# SPDX-License-Identifier: Apache-2.0
#

"""
MELODIES-MONET statistics processing module.
Architected by Aero 🍃⚡ for the Pangeo ecosystem.
"""

import inspect
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from melodies_monet.plots import savefig

# Using monet_stats for Pangeo-optimized, Aero-compliant statistics
import monet_stats.contingency_metrics
import monet_stats.correlation_metrics
import monet_stats.efficiency_metrics
import monet_stats.error_metrics
import monet_stats.relative_metrics


# Dynamically discover all statistics from monet-stats
def _discover_stats() -> Dict[str, Any]:
    """
    Discover all uppercase statistical functions in monet-stats submodules.

    Returns
    -------
    dict
        A mapping of statistic names to their corresponding functions.
    """
    modules = [
        monet_stats.contingency_metrics,
        monet_stats.correlation_metrics,
        monet_stats.efficiency_metrics,
        monet_stats.error_metrics,
        monet_stats.relative_metrics,
    ]
    discovered = {}
    for mod in modules:
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            # We assume stats are uppercase and from the module itself (not imported)
            if name.isupper() and func.__module__ == mod.__name__:
                discovered[name] = func
    return discovered


_ALL_STATS = _discover_stats()

# Pre-defined mapping for friendly names
_STAT_FRIENDLY_NAMES = {
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
    stat_fullname_list = []
    for stat_id in stat_list:
        fullname = _STAT_FRIENDLY_NAMES.get(stat_id, stat_id)
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
    **kwargs: Any,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate statistics in a backend-agnostic manner.

    This function supports both NumPy-backed and Dask-backed Xarray DataArrays,
    as well as Pandas Series (via DataFrame). It avoids forced computes on
    Xarray objects and maintains laziness when possible by using monet-stats.

    All statistics available in monet-stats are supported.

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
    **kwargs : Any
        Additional arguments passed to the monet-stats function.

    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Calculated statistical value.
    """
    if stat is None:
        raise ValueError("The 'stat' parameter must be provided.")

    # 1. Extract observations and model data without forcing a compute on xarray
    if isinstance(df, pd.DataFrame):
        obs = df[obsvar].values
        mod = df[modvar].values
    elif isinstance(df, xr.Dataset):
        obs = df[obsvar]
        mod = df[modvar]
    else:
        # Assuming df is some form of array-like (e.g., DataArray)
        obs = df
        mod = df

    # 2. Determine the stat function and apply circular logic if requested
    target_stat = stat
    if wind:
        if not stat.startswith("WD"):
            target_stat = f"WD{stat}"
        # Special case for WDNMB_m in monet-stats
        if target_stat == "WDNMB":
            target_stat = "WDNMB_m"

    func = _ALL_STATS.get(target_stat)
    if func is None:
        # Fallback to non-WD version if WD version not found
        func = _ALL_STATS.get(stat)
        if func is None:
            print(f"Stat not found in monet-stats: {target_stat}")
            return np.nan

    # 3. Aero Protocol: Handle explicit dimensions for Xarray/Dask to avoid reductions errors
    calc_kwargs = kwargs.copy()
    sig = inspect.signature(func)

    if "axis" in sig.parameters:
        if (
            isinstance(obs, xr.DataArray)
            and "axis" not in calc_kwargs
            and "dim" not in calc_kwargs
        ):
            if obs.ndim == 1:
                calc_kwargs["axis"] = obs.dims[0]
            else:
                calc_kwargs["axis"] = tuple(obs.dims)
        elif "axis" not in calc_kwargs:
            calc_kwargs["axis"] = None

    # Handle paxis if required and missing
    if "paxis" in sig.parameters and "paxis" not in calc_kwargs:
        if isinstance(obs, xr.DataArray):
            calc_kwargs["paxis"] = obs.dims[0]
        else:
            calc_kwargs["paxis"] = 0

    # 4. Perform calculation
    value = func(obs, mod, **calc_kwargs)

    # 5. Scientific Hygiene: Update history if it's an xarray object
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
