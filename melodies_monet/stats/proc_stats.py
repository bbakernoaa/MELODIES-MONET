# SPDX-License-Identifier: Apache-2.0
#

# Simple MONET utility to calculate statistics from paired hdf file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import monet_stats
import inspect

def produce_stat_dict(stat_list, spaces=False):
    """Select statistics. Returns the full name of the statistic.
    If the statistic is not found in monet_stats, it returns the ID as is.

    Parameters
    ----------
    stat_list : list of strings
        List of statistic abbreviations specified in input yaml file
    spaces : boolean
        Whether to leave spaces in the string containing the full name (True)
        or remove spaces (False).

    Returns
    -------
    list of strings
        list of full names of the statistics

    """
    stat_fullname_list = []
    for stat_id in stat_list:
        try:
            func = getattr(monet_stats, stat_id)
            doc = func.__doc__
            if doc:
                fullname = doc.strip().split('\n')[0].strip().strip('.')
            else:
                fullname = stat_id
        except AttributeError:
            fullname = stat_id

        if not spaces:
            fullname = fullname.replace(" ", "_")
        stat_fullname_list.append(fullname)

    return stat_fullname_list


def calc(df, stat=None, obsvar=None, modvar=None, wind=False, **kwargs):
    """
    Calculate statistical metrics between model and observation data.
    Dynamically calls functions from monet_stats, preserving lazy objects.

    Parameters
    ----------
    df : pandas.DataFrame or xarray.Dataset
        Model/observation paired data.
    stat : str
        Abbreviation of the statistic to calculate (e.g., 'MB', 'RMSE').
    obsvar : str
        Column label of the observation variable.
    modvar : str
        Column label of the model variable.
    wind : bool, optional
        Whether the variable is wind, requiring special handling of directional values.
        Default is False.
    **kwargs : Any
        Additional arguments passed to the statistic function (e.g., threshold, minval).

    Returns
    -------
    float or xarray.DataArray
        The calculated statistical value.
    """
    obs = df[obsvar]
    mod = df[modvar]

    # Check for wind-specific stats in monet_stats
    if wind:
        wind_stat = f"WD{stat}"
        if hasattr(monet_stats, wind_stat):
            stat = wind_stat

    try:
        stat_func = getattr(monet_stats, stat)
        sig = inspect.signature(stat_func)

        call_kwargs = {'axis': None}
        if 'minval' in sig.parameters:
            call_kwargs['minval'] = kwargs.get('threshold', kwargs.get('minval', 0.0))
        if 'threshold' in sig.parameters:
            call_kwargs['threshold'] = kwargs.get('threshold', 0.0)

        # Add any other matching kwargs
        for param in sig.parameters:
            if param in kwargs and param not in call_kwargs:
                call_kwargs[param] = kwargs[param]

        value = stat_func(obs, mod, **call_kwargs)

        # If it's an xarray object and we want a scalar for the table,
        # we might need to compute it eventually, but we should stay lazy as long as possible.
        # However, MELODIES-MONET stats table usually expects a scalar.
        # For now, return the object and let the caller decide when to compute.
        return value

    except AttributeError:
        print(f"Stat not found in monet_stats: {stat}")
        return np.nan
    except Exception as e:
        print(f"Error calculating {stat}: {e}")
        return np.nan


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

    # If the dataframe contains lazy objects, we must compute them before plotting the table.
    # This is the point where we 'break' laziness for visualization purposes.

    # We copy to avoid modifying the original if it was used elsewhere
    plot_df = df.copy()
    stat_full_names = plot_df["Stat_FullName"].values.tolist()
    plot_df = plot_df.drop(columns=["Stat_FullName"])

    # Compute any xarray objects
    for col in plot_df.columns:
        for idx in plot_df.index:
            val = plot_df.loc[idx, col]
            if hasattr(val, 'compute'):
                computed_val = val.compute()
                if hasattr(computed_val, 'item'):
                    plot_df.loc[idx, col] = computed_val.item()
                else:
                    plot_df.loc[idx, col] = computed_val

    t = ax.table(
        cellText=plot_df.values,
        rowLabels=stat_full_names,
        colLabels=plot_df.columns,
        loc="center",
        edges=table_kwargs["edges"],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(table_kwargs["fontsize"])
    t.scale(table_kwargs["xscale"], table_kwargs["yscale"])
    plt.title(title, fontsize=table_kwargs["fontsize"] * 1.1, fontweight="bold")
    fig.tight_layout()

    from melodies_monet.plots import savefig
    savefig(outname + ".png", loc=1, logo_height=70)

    return
