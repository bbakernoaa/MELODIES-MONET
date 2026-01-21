# SPDX-License-Identifier: Apache-2.0
#

"""
MELODIES-MONET statistics processing module.
Architected by Aero 🍃⚡ for the Pangeo ecosystem.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from melodies_monet.plots import savefig

# Using monet_stats for Pangeo-optimized, Aero-compliant statistics
from monet_stats.correlation_metrics import (
    AC,
    E1,
    IOA,
    R2,
    RMSE,
    WDAC,
    WDIOA,
    WDRMSE,
    d1,
)
from monet_stats.error_metrics import (
    MB,
    MNB,
    MNE,
    MO,
    MP,
    MdnB,
    MdnNB,
    MdnNE,
    MdnO,
    MdnP,
    NMdnGE,
    NO,
    NOP,
    NP,
    RM,
    RMdn,
    STDO,
    STDP,
    WDMB,
    WDMdnB,
)
from monet_stats.relative_metrics import (
    FB,
    FE,
    ME,
    MdnE,
    NMB,
    NMdnB,
    NME,
    NMdnE,
    WDME,
    WDMdnE,
    WDNMB_m,
)


def produce_stat_dict(stat_list: List[str], spaces: bool = False) -> List[str]:
    """
    Select statistics and return their full names.

    Parameters
    ----------
    stat_list : list of str
        List of statistic abbreviations specified in input yaml file.
    spaces : bool, optional
        Whether to leave spaces in the string containing the full name (True)
        or remove spaces (False).

    Returns
    -------
    list of str
        List of full names of the statistics.
    """
    dict_stats_def = {
        "STDO": "Obs Standard Deviation",
        "STDP": "Mod Standard Deviation",
        "MNB": "Mean Normalized Bias (%)",
        "MNE": "Mean Normalized Gross Error (%)",
        "MdnNB": "Median Normalized Bias (%)",
        "MdnNE": "Median Normalized Gross Error (%)",
        "NMdnGE": "Normalized Median Gross Error (%)",
        "NO": "Obs Number",
        "NOP": "Pairs Number",
        "NP": "Mod Number",
        "MO": "Obs Mean",
        "MP": "Mod Mean",
        "MdnO": "Obs Median",
        "MdnP": "Mod Median",
        "RM": "Mean Ratio Obs/Mod",
        "RMdn": "Median Ratio Obs/Mod",
        "MB": "Mean Bias",
        "MdnB": "Median Bias",
        "NMB": "Normalized Mean Bias (%)",
        "NMdnB": "Normalized Median Bias (%)",
        "FB": "Fractional Bias (%)",
        "ME": "Mean Gross Error",
        "MdnE": "Median Gross Error",
        "NME": "Normalized Mean Error (%)",
        "NMdnE": "Normalized Median Error (%)",
        "FE": "Fractional Error (%)",
        "R2": "Coefficient of Determination (R2)",
        "RMSE": "Root Mean Square Error",
        "d1": "Modified Index of Agreement",
        "E1": "Modified Coefficient of Efficiency",
        "IOA": "Index of Agreement",
        "AC": "Anomaly Correlation",
    }
    stat_fullname_list = []
    for stat_id in stat_list:
        if stat_id in dict_stats_def:
            fullname = dict_stats_def[stat_id]
        else:
            # Fallback for stats not in the default dictionary
            fullname = stat_id
        if not spaces:
            fullname = fullname.replace(" ", "_")
        stat_fullname_list.append(fullname)
    return stat_fullname_list


def calc(
    df: Union[pd.DataFrame, xr.Dataset, xr.DataArray],
    stat: Optional[str] = None,
    obsvar: Optional[str] = None,
    modvar: Optional[str] = None,
    wind: bool = False,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate statistics in a backend-agnostic manner.

    This function supports both NumPy-backed and Dask-backed Xarray DataArrays,
    as well as Pandas Series (via DataFrame). It avoids forced computes on
    Xarray objects and maintains laziness when possible by using monet-stats.

    Parameters
    ----------
    df : pandas.DataFrame or xarray.Dataset or xarray.DataArray
        Model/obs paired data.
    stat : str, optional
        Abbreviation of the statistic to calculate.
    obsvar : str, optional
        Column or variable name of observations.
    modvar : str, optional
        Column or variable name of model results.
    wind : bool, optional
        Whether the variable is wind direction (requires circular stats).

    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Calculated statistical value.
    """
    # Extract observations and model data without forcing a compute on xarray
    if isinstance(df, pd.DataFrame):
        # For pandas, we use .values to avoid issues with some monet-stats
        # functions that don't perfectly handle pd.Series with np.ma
        obs = df[obsvar].values
        mod = df[modvar].values
    elif isinstance(df, xr.Dataset):
        obs = df[obsvar]
        mod = df[modvar]
    else:
        # Assuming df is some form of array-like (e.g., DataArray)
        obs = df
        mod = df

    # Aero Protocol: For Xarray/Dask, explicit dimension names are preferred over None
    # to avoid NotImplementedError in dask for certain reductions (like nanmedian).
    # If it is 1D, we use the single dimension name instead of a tuple.
    calc_axis = None
    if isinstance(obs, xr.DataArray):
        if obs.ndim == 1:
            calc_axis = obs.dims[0]
        else:
            calc_axis = tuple(obs.dims)

    # Map stat abbreviations to monet-stats functions
    # Note: monet-stats functions handle both numpy and xarray/dask backends
    stat_func_map = {
        "STDO": lambda o, m: STDO(o, m, axis=calc_axis),
        "STDP": lambda o, m: STDP(o, m, axis=calc_axis),
        "MNB": lambda o, m: MNB(o, m, axis=calc_axis),
        "MNE": lambda o, m: MNE(o, m, axis=calc_axis),
        "MdnNB": lambda o, m: MdnNB(o, m, axis=calc_axis),
        "MdnNE": lambda o, m: MdnNE(o, m, axis=calc_axis),
        "NMdnGE": lambda o, m: NMdnGE(o, m, axis=calc_axis),
        "NO": lambda o, m: NO(o, m, axis=calc_axis),
        "NOP": lambda o, m: NOP(o, m, axis=calc_axis),
        "NP": lambda o, m: NP(o, m, axis=calc_axis),
        "MO": lambda o, m: MO(o, m, axis=calc_axis),
        "MP": lambda o, m: MP(o, m, axis=calc_axis),
        "MdnO": lambda o, m: MdnO(o, m, axis=calc_axis),
        "MdnP": lambda o, m: MdnP(o, m, axis=calc_axis),
        "RM": lambda o, m: RM(o, m, axis=calc_axis),
        "RMdn": lambda o, m: RMdn(o, m, axis=calc_axis),
        "MB": lambda o, m: WDMB(o, m, axis=calc_axis)
        if wind
        else MB(o, m, axis=calc_axis),
        "MdnB": lambda o, m: WDMdnB(o, m, axis=calc_axis)
        if wind
        else MdnB(o, m, axis=calc_axis),
        "NMB": lambda o, m: WDNMB_m(o, m, axis=calc_axis)
        if wind
        else NMB(o, m, axis=calc_axis),
        "NMdnB": lambda o, m: NMdnB(o, m, axis=calc_axis),
        "FB": lambda o, m: FB(o, m, axis=calc_axis),
        "ME": lambda o, m: WDME(o, m, axis=calc_axis)
        if wind
        else ME(o, m, axis=calc_axis),
        "MdnE": lambda o, m: WDMdnE(o, m, axis=calc_axis)
        if wind
        else MdnE(o, m, axis=calc_axis),
        "NME": lambda o, m: NME(o, m, axis=calc_axis),
        "NMdnE": lambda o, m: NMdnE(o, m, axis=calc_axis),
        "FE": lambda o, m: FE(o, m, axis=calc_axis),
        "R2": lambda o, m: R2(o, m, axis=calc_axis),
        "RMSE": lambda o, m: WDRMSE(o, m, axis=calc_axis)
        if wind
        else RMSE(o, m, axis=calc_axis),
        "d1": lambda o, m: d1(o, m, axis=calc_axis),
        "E1": lambda o, m: E1(o, m, axis=calc_axis),
        "IOA": lambda o, m: WDIOA(o, m, axis=calc_axis)
        if wind
        else IOA(o, m, axis=calc_axis),
        "AC": lambda o, m: WDAC(o, m, axis=calc_axis)
        if wind
        else AC(o, m, axis=calc_axis),
    }

    if stat in stat_func_map:
        value = stat_func_map[stat](obs, mod)
    else:
        print(f"Stat not found: {stat}")
        value = np.nan

    # Scientific Hygiene: Update history if it's an xarray object
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        if (
            "history" not in value.attrs
            or f"Calculated {stat}" not in value.attrs["history"]
        ):
            history = f"Calculated {stat} using monet-stats"
            value.attrs["history"] = (
                f"{value.attrs.get('history', '')}\n{history}".strip()
            )

    return value


def create_table(
    df: pd.DataFrame,
    outname: str = "plot",
    title: str = "stats",
    out_table_kwargs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> None:
    """
    Calculates all of the specified statistics, save to csv file, and
    optionally save to a figure visualizing the table.

    Parameters
    ----------
    df : pandas.DataFrame
        model/obs pair data
    outname : str, optional
        file location and name of plot (do not include .png)
    title : str, optional
        Title to include on the figure visualizing the table
    out_table_kwargs : dict, optional
        Dictionary containing information to create the figure visualizing the
        table.
    debug : bool, optional
        Whether to plot interactively (True) or not (False). Flag for
        submitting jobs to supercomputer turn off interactive mode.

    Returns
    -------
    None
    """
    if not debug:
        plt.ioff()

    # Define defaults if not provided:
    out_table_def = dict(
        fontsize=16.0, xscale=1.2, yscale=1.2, figsize=[10, 7], edges="open"
    )
    if out_table_kwargs is not None:
        table_kwargs = {**out_table_def, **out_table_kwargs}
    else:
        table_kwargs = out_table_def

    # Create a table graphic
    fig, ax = plt.subplots(figsize=table_kwargs["figsize"])
    ax.axis("off")
    ax.axis("tight")

    rows = df["Stat_FullName"].values.tolist()
    plot_df = df.drop(columns=["Stat_FullName"])

    t = ax.table(
        cellText=plot_df.values,
        rowLabels=rows,
        colLabels=plot_df.columns,
        loc="center",
        edges=table_kwargs["edges"],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(table_kwargs["fontsize"])
    t.scale(table_kwargs["xscale"], table_kwargs["yscale"])
    plt.title(title, fontsize=table_kwargs["fontsize"] * 1.1, fontweight="bold")
    fig.tight_layout()
    savefig(outname + ".png", loc=1, logo_height=70)

    return
