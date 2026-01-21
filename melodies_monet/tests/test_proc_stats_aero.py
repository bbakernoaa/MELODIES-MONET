# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for the statistics processing module.
Following the Aero Protocol: backend-agnostic validation.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from melodies_monet.stats import proc_stats


def test_calc_all_stats_pandas():
    """
    Verify that all discovered stats from monet-stats can be executed via proc_stats.calc
    using a Pandas backend.
    """
    obs_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 3.2, 3.8, 5.5])
    df_pd = pd.DataFrame({"obs": obs_data, "mod": mod_data})

    # Discovery check
    all_stats = proc_stats._ALL_STATS.keys()
    assert len(all_stats) > 20

    for stat in all_stats:
        kwargs = {}
        # Handle special cases that require thresholds
        if stat in ["CSI", "ETS", "FAR", "FBI", "HSS", "POD", "TSS"]:
            kwargs = {"minval": 2.0}

        try:
            res = proc_stats.calc(df_pd, stat=stat, obsvar="obs", modvar="mod", **kwargs)
            # Basic execution check
            assert res is not None
        except Exception as e:
            pytest.fail(f"Stat {stat} failed on Pandas backend: {e}")


def test_calc_backend_agnostic_representative():
    """
    Aero Protocol Validation: Verify logic works identical for NumPy (Pandas)
    and Dask (Xarray) backends for a representative subset.
    """
    obs_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 3.2, 3.8, 5.5])

    df_pd = pd.DataFrame({"obs": obs_data, "mod": mod_data})
    ds_np = xr.Dataset({"obs": (["x"], obs_data), "mod": (["x"], mod_data)})
    ds_da = ds_np.chunk({"x": -1})

    # Representative stats that work well with dask and don't require extra kwargs
    stats_to_test = ["MB", "RMSE", "NMB", "R2", "IOA", "MNB", "MO", "NO", "STDO"]

    for stat in stats_to_test:
        res_pd = proc_stats.calc(df_pd, stat=stat, obsvar="obs", modvar="mod")
        res_np = proc_stats.calc(ds_np, stat=stat, obsvar="obs", modvar="mod")
        res_da = proc_stats.calc(ds_da, stat=stat, obsvar="obs", modvar="mod")

        np.testing.assert_allclose(res_pd, res_np, err_msg=f"Backend mismatch: PD vs NP for {stat}")

        if isinstance(res_da, xr.DataArray):
            assert hasattr(res_da.data, "dask"), f"Dask laziness broken for {stat}"
            res_da_val = res_da.compute()
        else:
            res_da_val = res_da

        np.testing.assert_allclose(res_np, res_da_val, err_msg=f"Backend mismatch: NP vs DA for {stat}")


def test_wind_logic_fallback():
    """
    Verify that setting wind=True correctly routes to WD versions.
    """
    obs_wind = np.array([350.0, 10.0])
    mod_wind = np.array([10.0, 350.0])
    df = pd.DataFrame({"obs": obs_wind, "mod": mod_wind})

    # Should use WDMB
    res = proc_stats.calc(df, stat="MB", obsvar="obs", modvar="mod", wind=True)
    np.testing.assert_allclose(float(res), 0.0, atol=1e-7)


def test_provenance():
    """Verify history attribute."""
    ds = xr.Dataset({"obs": (["x"], [1, 2]), "mod": (["x"], [1.1, 1.9])})
    res = proc_stats.calc(ds, stat="RMSE", obsvar="obs", modvar="mod")
    assert "history" in res.attrs
    assert "Calculated RMSE using monet-stats" in res.attrs["history"]


def test_produce_stat_dict():
    """Verify discovery in produce_stat_dict."""
    stats = ["COE", "MB"]
    names = proc_stats.produce_stat_dict(stats, spaces=True)
    assert names[0] == "COE"
    assert names[1] == "Mean Bias"
