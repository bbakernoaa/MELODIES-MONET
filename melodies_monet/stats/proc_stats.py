# SPDX-License-Identifier: Apache-2.0
#

# Simple MONET utility to calculate statistics from paired hdf file

import inspect

import matplotlib.pyplot as plt
import monet_stats.contingency_metrics as co
import monet_stats.correlation_metrics as cm
import monet_stats.efficiency_metrics as ef
import monet_stats.error_metrics as em
import monet_stats.relative_metrics as rm
import monet_stats.spatial_ensemble_metrics as sem
import monet_stats.spatial_skill_metrics as ssm
import numpy as np
import pandas as pd
import xarray as xr

from melodies_monet.plots import savefig

# Discover all available statistics from monet-stats
_STAT_FUNCS = {}
for mod in [em, rm, cm, ef, co, sem, ssm]:
    for name, func in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("_"):
            _STAT_FUNCS[name] = func


def produce_stat_dict(stat_list, spaces=False):
    """Select statistics.

    Parameters
    ----------
    stat_list : list of strings
        List of statistic abbreviations specified in input yaml file
    spaces : boolean
        Whether to leave spaces in the string containing the full name (True)
        or remove spaces (False).

    Returns
    -------
    list
        list of statistics full names

    """
    # Legacy mapping for backward compatibility and preferred naming
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
            name = dict_stats_def[stat_id]
        elif stat_id in _STAT_FUNCS:
            # Dynamically get name from docstring if available
            doc = _STAT_FUNCS[stat_id].__doc__
            if doc:
                lines = [line.strip() for line in doc.split("\n") if line.strip()]
                name = lines[0] if lines else stat_id
            else:
                name = stat_id
        else:
            name = stat_id

        if not spaces:
            stat_fullname_list.append(name.replace(" ", "_"))
        else:
            stat_fullname_list.append(name)

    return stat_fullname_list


def calc(df, stat=None, obsvar=None, modvar=None, wind=False):
    """Calculate statistics

    Parameters
    ----------
    df : dataframe
        model/obs pair data
    stat : str
        Statistic abbreviation
    obsvar : str
        Column label of observation variable
    modvar : str
        Column label of model variable
    wind : bool
        If variable is wind MONET applies a special calculation to handle
        negative and positive values. If wind (True) if not wind (False).

    Returns
    -------
    real
        statistical value

    """
    obs = df[obsvar]
    mod = df[modvar]

    # Aero Protocol: Keep xarray/dask objects as is, but convert pandas to numpy
    # because some monet-stats functions use np.ma functions that don't handle
    # pandas Series well.
    if isinstance(obs, pd.Series):
        obs = obs.values
    if isinstance(mod, pd.Series):
        mod = mod.values

    func = None
    if wind:
        # Priority 1: Check for special wind naming in monet-stats
        wd_stat = "WD" + stat
        if wd_stat in _STAT_FUNCS:
            func = _STAT_FUNCS[wd_stat]
        elif stat == "NMB":  # Special case for legacy WDNMB_m
            if "WDNMB_m" in _STAT_FUNCS:
                func = _STAT_FUNCS["WDNMB_m"]

    if func is None:
        if stat in _STAT_FUNCS:
            func = _STAT_FUNCS[stat]

    if func is not None:
        # Most monet-stats functions take (obs, mod, axis=None)
        # We use axis=None to compute over the whole flattened data
        try:
            value = func(obs, mod, axis=None)
        except TypeError:
            # Some might not take axis
            try:
                value = func(obs, mod)
            except Exception as e:
                print(f"Error calling {stat}: {e}")
                value = np.nan
    else:
        print("Stat not found: " + str(stat))
        value = np.nan

    # If the result is still a dask array or xarray object, compute/load it
    # since this function is expected to return a scalar value in the current driver.
    if hasattr(value, "compute"):
        value = value.compute()
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        value = value.values

    return value


def create_table(df, outname="plot", title="stats", out_table_kwargs=None, debug=False):
    """Calculates all of the specified statistics, save to csv file, and
    optionally save to a figure visualizing the table.

    Parameters
    ----------
    df : dataframe
        model/obs pair data
    outname : str
        file location and name of plot (do not include .png)
    title : str
        Title to include on the figure visualizing the table
    out_table_kwargs : dictionary
        Dictionary containing information to create the figure visualizing the
        table.
    debug : boolean
        Whether to plot interactively (True) or not (False). Flag for
        submitting jobs to supercomputer turn off interactive mode.

    Returns
    -------
    csv file, plot
        csv file and optional plot containing the statistical calculations
        specified in the input yaml file.

    """
    if debug is False:
        plt.ioff()

    # Define defaults if not provided:
    out_table_def = dict(fontsize=16.0, xscale=1.2, yscale=1.2, figsize=[10, 7], edges="open")
    if out_table_kwargs is not None:
        table_kwargs = {**out_table_def, **out_table_kwargs}
    else:
        table_kwargs = out_table_def

    # Create a table graphic
    fig, ax = plt.subplots(figsize=table_kwargs["figsize"])
    ax.axis("off")
    ax.axis("tight")

    rows = df["Stat_FullName"].values.tolist()

    df = df.drop(columns=["Stat_FullName"])

    t = ax.table(
        cellText=df.values,
        rowLabels=rows,
        colLabels=df.columns,
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
