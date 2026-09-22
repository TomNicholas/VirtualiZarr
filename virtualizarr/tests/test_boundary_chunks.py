"""How each archival format stores a boundary chunk, and what that means for
virtual concatenation.

Every format covered here except TIFF strips stores a boundary chunk at its full
declared chunk size, padding the unused region on disk. That padding is harmless
while the chunk stays at the end of an axis, because the array's shape crops it
off. It stops being harmless if virtual concatenation puts more data after it,
since a chunk grid has no way to say "this chunk decodes to more elements than it
contributes".
"""

import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import (
    ManifestArray,
    ManifestGroup,
    ManifestStore,
    array_api,
)
from virtualizarr.manifests.utils import copy_and_replace_metadata
from virtualizarr.parsers import HDFParser, ZarrParser
from virtualizarr.tests import requires_icechunk, requires_tiff, requires_tifffile
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX

DTYPE = np.dtype("uint16")
# TIFF requires tile dimensions to be multiples of 16, so every format here uses
# the same chunk size to keep the cases comparable
CHUNKS = (16, 16)


def _write_netcdf(path: Path, data: np.ndarray, chunks: tuple[int, int] = CHUNKS):
    xr.Dataset({"v": (("y", "x"), data)}).to_netcdf(
        path.with_suffix(".nc"),
        encoding={"v": {"chunksizes": chunks}},
        engine="h5netcdf",
    )
    return f"file://{path.with_suffix('.nc')}", HDFParser(), "v"


def _write_zarr(path: Path, data: np.ndarray, chunks: tuple[int, int] = CHUNKS):
    group = zarr.create_group(str(path.with_suffix(".zarr")), overwrite=True)
    arr = group.create_array(
        "v",
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        dimension_names=("y", "x"),
        compressors=None,
    )
    arr[:] = data
    return f"file://{path.with_suffix('.zarr')}", ZarrParser(), "v"


def _write_tiff_tiled(path: Path, data: np.ndarray, chunks: tuple[int, int] = CHUNKS):
    import tifffile
    import virtual_tiff

    tifffile.imwrite(str(path.with_suffix(".tif")), data, tile=chunks)
    return f"file://{path.with_suffix('.tif')}", virtual_tiff.VirtualTIFF(ifd=0), "0"


@pytest.fixture(
    params=[
        pytest.param(_write_netcdf, id="netcdf4"),
        pytest.param(_write_zarr, id="zarr"),
        pytest.param(
            _write_tiff_tiled, id="tiff-tiled", marks=[requires_tiff, requires_tifffile]
        ),
    ]
)
def writer(request):
    """A callable (path, data) -> (url, parser, variable name)."""
    return request.param


def _read_back(var: xr.Variable, registry) -> np.ndarray:
    """Materialize a virtual variable's values through a ManifestStore.

    xarray consumes `dimension_names` off the metadata when it opens a store, so
    they have to be put back before the array can be served from a fresh one.
    """
    marr = var.data
    marr = ManifestArray(
        metadata=copy_and_replace_metadata(
            marr.metadata, new_dimension_names=[str(d) for d in var.dims]
        ),
        chunkmanifest=marr.manifest,
    )
    store = ManifestStore(group=ManifestGroup(arrays={"v": marr}), registry=registry)
    with xr.open_zarr(store, consolidated=False, zarr_format=3) as ds:
        return ds["v"].values


def test_zarr_requires_a_chunk_to_decode_to_its_declared_shape(tmp_path):
    """The assumption the rest of this module rests on.

    zarr decodes a chunk and reshapes it to the shape the grid declares, with no
    tolerance in either direction. Swapping one interior chunk for a compressed
    blob that inflates to more elements than the grid declares makes the array
    unreadable - and only once that chunk is actually fetched, so a store can stay
    corrupt and unnoticed until something touches the wrong region.
    """
    store = tmp_path / "a.zarr"
    arr = zarr.create_array(
        str(store),
        shape=(550,),
        chunks=(200,),
        dtype=DTYPE,
        compressors=zarr.codecs.ZstdCodec(),
    )
    arr[:] = np.arange(550)

    # a compressed blob that decodes to 300 elements rather than the declared 200
    donor = tmp_path / "donor.zarr"
    donor_arr = zarr.create_array(
        str(donor),
        shape=(300,),
        chunks=(300,),
        dtype=DTYPE,
        compressors=zarr.codecs.ZstdCodec(),
    )
    donor_arr[:] = np.arange(1000, 1300)
    shutil.copyfile(donor / "c" / "0", store / "c" / "1")

    reopened = zarr.open_array(str(store), mode="r")
    # chunk 0 is untouched, so that region still reads back correctly
    np.testing.assert_array_equal(reopened[0:200], np.arange(200, dtype=DTYPE))
    with pytest.raises(ValueError, match="cannot reshape array of size 300"):
        reopened[:]


def test_boundary_chunk_is_stored_padded(writer, tmp_path, local_registry):
    """A shape that isn't a multiple of the chunk size stores its boundary chunks
    at the full chunk size, padded."""
    # 24x24 chunked 16x16 -> a 2x2 grid in which only chunk (0,0) is entirely
    # real; the others hold 8 real rows and/or columns but are stored as 16x16
    data = np.arange(24 * 24, dtype=DTYPE).reshape(24, 24)
    url, parser, name = writer(tmp_path / "a", data)

    vds = open_virtual_dataset(url, parser=parser, registry=local_registry)
    entries = vds[name].data.manifest.dict()

    full_chunk_nbytes = CHUNKS[0] * CHUNKS[1] * DTYPE.itemsize
    assert len(entries) == 4, "expected a 2x2 grid of chunks"
    for key, entry in entries.items():
        assert entry["length"] == full_chunk_nbytes, (
            f"chunk {key} is {entry['length']} bytes rather than the full "
            f"{full_chunk_nbytes}: this format does not pad its boundary chunks, "
            "so the assumptions in this module need revisiting"
        )


def test_boundary_chunk_padding_is_cropped_by_shape(writer, tmp_path, local_registry):
    """The padding is invisible on read, because the array's shape crops it off."""
    data = np.arange(24 * 24, dtype=DTYPE).reshape(24, 24)
    url, parser, name = writer(tmp_path / "a", data)

    vds = open_virtual_dataset(url, parser=parser, registry=local_registry)
    np.testing.assert_array_equal(_read_back(vds[name].variable, local_registry), data)


def test_concat_chunk_aligned_reads_back_correctly(writer, tmp_path, local_registry):
    """Concatenating along an axis whose length IS a multiple of the chunk size.

    No boundary chunk ends up in the interior, so the result stays an ordinary
    regular chunk grid and every value survives the round trip.
    """
    top = np.arange(32 * 16, dtype=DTYPE).reshape(32, 16)
    bottom = np.arange(5000, 5000 + 32 * 16, dtype=DTYPE).reshape(32, 16)
    url1, parser, name = writer(tmp_path / "a", top)
    url2, _, _ = writer(tmp_path / "b", bottom)

    vds1 = open_virtual_dataset(url1, parser=parser, registry=local_registry)
    vds2 = open_virtual_dataset(url2, parser=parser, registry=local_registry)
    combined = xr.concat([vds1, vds2], dim="y")

    assert combined[name].shape == (64, 16)
    np.testing.assert_array_equal(
        _read_back(combined[name].variable, local_registry),
        np.concatenate([top, bottom], axis=0),
    )


def test_concat_with_interior_boundary_chunk_raises(writer, tmp_path, local_registry):
    """Concatenating along an axis whose length is NOT a multiple of the chunk size.

    The first file's final row of chunks is padded, and concatenation would place
    real data after it. A regular chunk grid cannot describe that, so this is
    rejected rather than silently reading the padding back as data.
    """
    top = np.arange(24 * 16, dtype=DTYPE).reshape(24, 16)  # 24 rows, chunked at 16
    bottom = np.arange(5000, 5000 + 24 * 16, dtype=DTYPE).reshape(24, 16)
    url1, parser, name = writer(tmp_path / "a", top)
    url2, _, _ = writer(tmp_path / "b", bottom)

    vds1 = open_virtual_dataset(url1, parser=parser, registry=local_registry)
    vds2 = open_virtual_dataset(url2, parser=parser, registry=local_registry)

    with pytest.raises(ValueError, match="partial chunk"):
        xr.concat([vds1, vds2], dim="y")


@requires_icechunk
@pytest.mark.parametrize(
    "writer, chunk_pair, expected_edges",
    [
        pytest.param(
            _write_netcdf, ((16, 16), (12, 16)), (16, 8, 12, 12), id="netcdf4"
        ),
        pytest.param(_write_zarr, ((16, 16), (12, 16)), (16, 8, 12, 12), id="zarr"),
        # TIFF tile dimensions must be multiples of 16, so the second file uses a
        # tile taller than the image rather than 12: its single tile holds 24 real
        # rows padded out to 32
        pytest.param(
            _write_tiff_tiled,
            ((16, 16), (32, 16)),
            (16, 8, 24),
            id="tiff-tiled",
            marks=[requires_tiff, requires_tifffile],
        ),
    ],
)
def test_concat_past_a_padded_chunk_writes_an_unreadable_store(
    writer, chunk_pair, expected_edges, tmp_path
):
    """Why the partial-chunk guard cannot simply be relaxed once rectilinear chunk
    grids are enabled.

    Concatenation promotes to a rectilinear grid when the inputs declare different
    chunk sizes along the concat axis. The promoted grid records each chunk's edge
    as the count it *contributes* - 8 rows for the first file's boundary chunk -
    but that chunk is still stored padded out to its full declared size. The edge
    lengths line up with the manifest entries, so nothing structural objects and
    Icechunk commits the store; the mismatch only surfaces on read.

    See https://github.com/zarr-developers/zarr-extensions/issues/74.
    """
    import icechunk

    top = np.arange(1, 24 * 16 + 1, dtype=DTYPE).reshape(24, 16)
    bottom = np.arange(5000, 5000 + 24 * 16, dtype=DTYPE).reshape(24, 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    vdss = []
    for path_name, data, chunks in [
        ("top", top, chunk_pair[0]),
        ("bottom", bottom, chunk_pair[1]),
    ]:
        url, parser, name = writer(tmp_path / path_name, data, chunks)
        vdss.append(open_virtual_dataset(url, parser=parser, registry=registry))

    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            url_prefix=PYTEST_TMP_DIRECTORY_URL_PREFIX,
            store=icechunk.local_filesystem_store(PYTEST_TMP_DIRECTORY_URL_PREFIX),
        )
    )
    repo = icechunk.Repository.create(
        storage=icechunk.Storage.new_in_memory(),
        config=config,
        authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
    )

    with zarr.config.set({"array.rectilinear_chunks": True}):
        # concatenating is refused precisely because of the padded boundary chunk
        with pytest.raises(ValueError, match="partial chunk"):
            xr.concat(vdss, dim="y")

        # bypass that guard to reach the state it is protecting against
        with mock.patch.object(
            array_api, "check_no_partial_chunks_on_concat_axis", lambda *a, **k: None
        ):
            combined = xr.concat(vdss, dim="y")

        # one edge length per manifest entry, so the metadata is self-consistent
        marr = combined[name].data
        assert marr.metadata.chunk_grid.chunk_shapes[0] == expected_edges
        assert marr.manifest.shape_chunk_grid[0] == len(expected_edges)

        session = repo.writable_session("main")
        combined.vz.to_icechunk(session.store)
        session.commit("write a store whose padded chunk is no longer last")

        # a chunk declared as contributing fewer rows still decodes to its full
        # padded size, so the store cannot be read back
        arr = zarr.open_array(
            store=repo.readonly_session("main").store,
            path=name,
            mode="r",
            zarr_format=3,
        )
        with pytest.raises(ValueError, match="cannot reshape array"):
            arr[:]


@requires_tiff
@requires_tifffile
def test_tiff_strips_are_not_padded(tmp_path):
    """The counterexample: a TIFF strip holds only its real rows.

    tifffile writes a short final strip rather than padding it, so unlike every
    other format here its boundary chunk's encoded extent equals its logical one.
    """
    import tifffile

    tifffile.imwrite(
        str(tmp_path / "s.tif"), np.ones((100, 50), dtype="uint8"), rowsperstrip=30
    )
    with tifffile.TiffFile(tmp_path / "s.tif") as tf:
        byte_counts = tf.pages[0].tags["StripByteCounts"].value

    assert byte_counts == (1500, 1500, 1500, 500)
    assert byte_counts[-1] < byte_counts[0], "final strip should be short, not padded"
