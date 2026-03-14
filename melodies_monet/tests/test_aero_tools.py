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


def test_wsdir2uv_parity():
    """Verify wsdir2uv works identically for NumPy and Dask backends."""
    ws_data = np.array([10.0, 5.0, 0.0])
    wdir_data = np.array([0.0, 90.0, 180.0])

    ws_eager = xr.DataArray(ws_data, dims="x", name="ws")
    wdir_eager = xr.DataArray(wdir_data, dims="x", name="wdir")

    # Eager execution
    u_eager, v_eager = tools.wsdir2uv(ws_eager, wdir_eager)

    # Lazy execution
    ws_lazy = ws_eager.chunk({"x": 1})
    wdir_lazy = wdir_eager.chunk({"x": 1})
    u_lazy, v_lazy = tools.wsdir2uv(ws_lazy, wdir_lazy)

    # Assertions
    xr.testing.assert_allclose(u_eager, u_lazy.compute())
    xr.testing.assert_allclose(v_eager, v_lazy.compute())
    assert u_eager.attrs["long_name"] == "u_wind_component"
    assert v_eager.attrs["long_name"] == "v_wind_component"
    assert u_lazy.chunks is not None


def test_get_relhum_parity():
    """Verify get_relhum works identically for NumPy and Dask backends."""
    temp_data = np.array([290.0, 300.0])
    press_data = np.array([101325.0, 100000.0])
    vap_data = np.array([0.01, 0.02])

    temp_eager = xr.DataArray(temp_data, dims="x", name="temp")
    press_eager = xr.DataArray(press_data, dims="x", name="press")
    vap_eager = xr.DataArray(vap_data, dims="x", name="vap")

    # Eager execution
    rh_eager = tools.get_relhum(temp_eager, press_eager, vap_eager)

    # Lazy execution
    temp_lazy = temp_eager.chunk({"x": 1})
    press_lazy = press_eager.chunk({"x": 1})
    vap_lazy = vap_eager.chunk({"x": 1})
    rh_lazy = tools.get_relhum(temp_lazy, press_lazy, vap_lazy)

    # Assertions
    xr.testing.assert_allclose(rh_eager, rh_lazy.compute())
    assert rh_eager.attrs["units"] == "%"
    assert "history" in rh_eager.attrs
    assert rh_lazy.chunks is not None
