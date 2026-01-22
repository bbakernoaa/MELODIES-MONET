# SPDX-License-Identifier: Apache-2.0
import pytest
import xarray as xr
import numpy as np
import dask.array as da
from melodies_monet.util import driver_util

def test_apply_mask_and_scale():
    # Setup eager data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ds_eager = xr.Dataset({"v1": (["x"], data)})

    # Setup lazy data
    ds_lazy = xr.Dataset({"v1": (["x"], da.from_array(data, chunks=2))})

    vdict = {
        "v1": {
            "obs_min": 2.0,
            "obs_max": 4.0,
            "unit_scale": 2.0,
            "unit_scale_method": "*"
        }
    }

    res_eager = driver_util.apply_mask_and_scale(ds_eager.copy(), vdict)
    res_lazy = driver_util.apply_mask_and_scale(ds_lazy.copy(), vdict)

    expected = np.array([np.nan, 4.0, 6.0, 8.0, np.nan])

    np.testing.assert_allclose(res_eager["v1"].values, expected)
    assert isinstance(res_lazy["v1"].data, da.Array)
    np.testing.assert_allclose(res_lazy["v1"].compute().values, expected)
    assert "history" in res_eager.attrs

def test_apply_variable_rename():
    ds = xr.Dataset({"old_v1": (["x"], [1, 2, 3])})
    vdict = {"old_v1": {"rename": "new_v1"}}

    res = driver_util.apply_variable_rename(ds, vdict)

    assert "new_v1" in res.data_vars
    assert "old_v1" not in res.data_vars
    assert "new_v1" in vdict # Check that vdict was updated
    assert "history" in res.attrs

def test_apply_variable_summing():
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])
    ds_lazy = xr.Dataset({
        "v1": (["x"], da.from_array(data1, chunks=2)),
        "v2": (["x"], da.from_array(data2, chunks=2))
    })

    vsum = {"v_total": {"vars": ["v1", "v2"]}}

    res = driver_util.apply_variable_summing(ds_lazy, vsum)

    assert "v_total" in res.data_vars
    assert isinstance(res["v_total"].data, da.Array)
    np.testing.assert_allclose(res["v_total"].compute().values, [5, 7, 9])
    assert "history" in res.attrs
