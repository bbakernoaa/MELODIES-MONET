# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for the statistics processing module.
Following the Aero Protocol: backend-agnostic validation.
"""

import numpy as np
import pandas as pd
import xarray as xr
from melodies_monet.stats import proc_stats


def test_calc_backend_agnostic():
    """
    Verify that proc_stats.calc produces identical results for NumPy (Pandas)
    and Dask (Xarray) backends.
    """
    # 1. Setup sample data
    obs_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 3.2, 3.8, 5.5])

    # Pandas backend (standard for surface data)
    df_pd = pd.DataFrame({"obs": obs_data, "mod": mod_data})

    # Xarray NumPy backend
    ds_np = xr.Dataset({"obs": (["x"], obs_data), "mod": (["x"], mod_data)})

    # Xarray Dask backend
    ds_da = ds_np.chunk({"x": -1})

    # 2. Representative set of statistics to test
    stats_to_test = [
        "MB",
        "RMSE",
        "NMB",
        "R2",
        "IOA",
        "MNB",
        "MNE",
        "MO",
        "MP",
        "NO",
        "NP",
        "STDO",
        "STDP",
    ]

    for stat in stats_to_test:
        # Calculate with Pandas
        res_pd = proc_stats.calc(df_pd, stat=stat, obsvar="obs", modvar="mod")

        # Calculate with Xarray NumPy
        res_np = proc_stats.calc(ds_np, stat=stat, obsvar="obs", modvar="mod")

        # Calculate with Xarray Dask
        res_da = proc_stats.calc(ds_da, stat=stat, obsvar="obs", modvar="mod")

        # 3. Assertions
        # Compare Pandas vs Xarray NumPy
        np.testing.assert_allclose(
            res_pd, res_np, err_msg=f"Backend mismatch: PD vs NP for {stat}"
        )

        # Compare Xarray NumPy vs Xarray Dask
        if isinstance(res_da, xr.DataArray):
            # Ensure laziness is preserved: result should be a dask array
            assert hasattr(res_da.data, "dask"), f"Dask laziness broken for {stat}"
            res_da_val = res_da.compute()
        else:
            res_da_val = res_da

        np.testing.assert_allclose(
            res_np, res_da_val, err_msg=f"Backend mismatch: NP vs DA for {stat}"
        )


def test_calc_median_stats_numpy():
    """
    Verify median-based stats on NumPy backends.
    Note: Skip Dask for global reductions due to known nanmedian limitations.
    """
    obs_data = np.array([1.0, 2.0, 10.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 8.0, 3.8, 5.5])

    ds_np = xr.Dataset({"obs": (["x"], obs_data), "mod": (["x"], mod_data)})

    median_stats = ["NMdnGE", "MdnE", "MdnNB", "MdnNE", "MdnO", "MdnP", "MdnB", "NMdnB"]

    for stat in median_stats:
        res = proc_stats.calc(ds_np, stat=stat, obsvar="obs", modvar="mod")
        assert not np.isnan(res), f"Median stat {stat} returned NaN"


def test_wind_direction_stats():
    """
    Verify circular statistics for wind direction.
    """
    # monet-stats MB(obs, mod) = mean(obs - mod) with circular correction
    obs_wind = np.array([350.0, 10.0])
    mod_wind = np.array([10.0, 350.0])

    ds = xr.Dataset({"obs": (["x"], obs_wind), "mod": (["x"], mod_wind)})

    # 350 -> 10: obs - mod = 350 - 10 = 340 -> -20 (overestimated by model if we consider 350 to 10 as +20)
    # Actually monet-stats:
    # circlebias(350 - 10) = circlebias(340) = -20
    # circlebias(10 - 350) = circlebias(-340) = +20
    # Mean = 0

    res = proc_stats.calc(ds, stat="MB", obsvar="obs", modvar="mod", wind=True)
    np.testing.assert_allclose(float(res), 0.0, atol=1e-7)


def test_provenance_tracking():
    """
    Verify that transformation history is tracked in Xarray objects.
    """
    ds = xr.Dataset({"obs": (["x"], [1, 2]), "mod": (["x"], [1.1, 1.9])})
    res = proc_stats.calc(ds, stat="MB", obsvar="obs", modvar="mod")

    assert "history" in res.attrs
    assert "Calculated MB using monet-stats" in res.attrs["history"]


def test_produce_stat_dict():
    """
    Verify the utility function for mapping stat abbreviations to names.
    """
    stats = ["MB", "RMSE", "UNKNOWN"]

    # Default (no spaces)
    names = proc_stats.produce_stat_dict(stats, spaces=False)
    assert names == ["Mean_Bias", "Root_Mean_Square_Error", "UNKNOWN"]

    # With spaces
    names = proc_stats.produce_stat_dict(stats, spaces=True)
    assert names == ["Mean Bias", "Root Mean Square Error", "UNKNOWN"]
