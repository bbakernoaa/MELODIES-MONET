import numpy as np
import pandas as pd
import xarray as xr

from melodies_monet.util import tools


def test_calc_geolocaltime_parity():
    """Verify calc_geolocaltime produces identical results for NumPy and Dask backends."""
    times = pd.date_range("2023-01-01", periods=3, freq="h")
    lons = np.array([-120, 0, 120])

    ds_numpy = xr.Dataset({"longitude": (("x",), lons)}, coords={"time": (("x",), times)})

    ds_dask = ds_numpy.chunk({"x": 1})

    res_numpy = tools.calc_geolocaltime(ds_numpy)
    res_dask = tools.calc_geolocaltime(ds_dask)

    # Assert lazy execution for dask backend
    assert res_dask.chunks is not None

    # Assert bit-for-bit parity
    xr.testing.assert_allclose(res_numpy, res_dask.compute())


def test_column_calculations_parity():
    """Verify partial and total column calculations produce identical results for NumPy and Dask."""
    # Create mock dataset
    nz, nx = 5, 10
    ds = xr.Dataset(
        {
            "NO2": (("z", "x"), np.random.rand(nz, nx)),
            "pres_pa_mid": (("z", "x"), np.linspace(101325, 10000, nz)[:, np.newaxis] * np.ones((1, nx))),
            "dz_m": (("z", "x"), np.ones((nz, nx)) * 100),
            "temperature_k": (("z", "x"), np.ones((nz, nx)) * 288),
            "surfpres_pa": (("x",), np.ones(nx) * 101325),
        }
    )

    ds_dask = ds.chunk({"x": 2})

    # Partial Column
    pcol_numpy = tools.calc_partialcolumn(ds, var="NO2")
    pcol_dask = tools.calc_partialcolumn(ds_dask, var="NO2")

    assert pcol_dask.chunks is not None
    xr.testing.assert_allclose(pcol_numpy, pcol_dask.compute())

    # Total Column
    tcol_numpy = tools.calc_totalcolumn(ds, var="NO2")
    tcol_dask = tools.calc_totalcolumn(ds_dask, var="NO2")

    assert tcol_dask.chunks is not None
    xr.testing.assert_allclose(tcol_numpy, tcol_dask.compute())


def test_provenance_update():
    """Verify that history attribute is updated."""
    ds = xr.Dataset({"longitude": (("x",), [0]), "time": (("x",), [pd.Timestamp("2023-01-01")])})
    ds.attrs["history"] = "Original history."

    _ = tools.calc_geolocaltime(ds)

    assert "Calculated geolocaltime" in ds.attrs["history"]
    assert "Original history." in ds.attrs["history"]
