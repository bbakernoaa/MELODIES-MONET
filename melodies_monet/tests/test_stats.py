# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from melodies_monet.stats import proc_stats


def test_calc_numpy_vs_dask():
    # Create sample data
    obs_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 3.2, 3.8, 5.5])

    # 1. Test with Pandas (which is what melodies-monet currently uses)
    df = pd.DataFrame({"obs": obs_data, "mod": mod_data})
    res_mb_pd = proc_stats.calc(df, stat="MB", obsvar="obs", modvar="mod")
    res_rmse_pd = proc_stats.calc(df, stat="RMSE", obsvar="obs", modvar="mod")
    assert not np.isnan(res_rmse_pd)

    # 2. Test with Xarray (NumPy backed)
    ds_np = xr.Dataset({"obs": (("x",), obs_data), "mod": (("x",), mod_data)})
    res_mb_xr_np = proc_stats.calc(ds_np, stat="MB", obsvar="obs", modvar="mod")

    # 3. Test with Xarray (Dask backed)
    ds_da = xr.Dataset(
        {
            "obs": (("x",), da.from_array(obs_data, chunks=2)),
            "mod": (("x",), da.from_array(mod_data, chunks=2)),
        }
    )
    res_mb_xr_da = proc_stats.calc(ds_da, stat="MB", obsvar="obs", modvar="mod")

    # Verify they are all the same
    assert np.isclose(res_mb_pd, -0.1)
    assert np.isclose(res_mb_pd, res_mb_xr_np)
    # For dask, monet-stats might return a dask array or a computed value depending on the implementation
    # If it returns a dask array, we need to compute it.
    if hasattr(res_mb_xr_da, "compute"):
        res_mb_xr_da = res_mb_xr_da.compute()
    assert np.isclose(res_mb_pd, res_mb_xr_da)

    # Test a few other stats to ensure dynamic discovery works
    res_mae = proc_stats.calc(df, stat="MAE", obsvar="obs", modvar="mod")
    assert np.isclose(res_mae, 0.22)

    # Test dynamic discovery from monet-stats that wasn't in legacy
    # e.g., 'KGE' (Kling-Gupta Efficiency)
    res_kge = proc_stats.calc(df, stat="KGE", obsvar="obs", modvar="mod")
    assert not np.isnan(res_kge)


def test_produce_stat_dict():
    stat_list = ["MB", "RMSE", "KGE"]
    names = proc_stats.produce_stat_dict(stat_list, spaces=True)
    assert "Mean Bias" in names
    assert "Root Mean Square Error" in names
    # KGE should have been discovered from monet-stats
    assert any("Kling-Gupta" in n for n in names) or "KGE" in names


def test_wind_stats():
    # Sample wind data (degrees)
    obs_wd = np.array([350, 10])
    mod_wd = np.array([10, 350])
    df = pd.DataFrame({"obs": obs_wd, "mod": mod_wd})

    # Standard MB would give (350-10 + 10-350)/2 = 0
    # But wind MB should handle the wrap around.
    # 350 to 10 is +20 degrees. 10 to 350 is -20 degrees.
    # Wait, monet-stats WDMB:
    # circlebias(350, 10) -> 20
    # circlebias(10, 350) -> -20
    # Mean is 0.

    res_wdmb = proc_stats.calc(df, stat="MB", obsvar="obs", modvar="mod", wind=True)
    assert not np.isnan(res_wdmb)

    # Test NMB wind (WDNMB_m in monet-stats)
    res_wdnmb = proc_stats.calc(df, stat="NMB", obsvar="obs", modvar="mod", wind=True)
    assert not np.isnan(res_wdnmb)
