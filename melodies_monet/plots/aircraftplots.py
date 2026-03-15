# SPDX-License-Identifier: Apache-2.0
#
"""
aircraftplots.py
this code is designed for plotting aircraft specific
plot types
"""

import math

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import monet as monet
import monet_plots
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from monet.util.tools import get_epa_region_bounds as get_epa_bounds

from melodies_monet.plots import savefig

sns.set_context("paper")


# Define a custom formatting function
def custom_yaxis_formatter(x, pos):
    if x == int(x):
        return str(int(x))
    else:
        return str(x)


def make_spatial_bias(
    df: pd.DataFrame,
    df_reg: pd.DataFrame = None,
    column_o: str = None,
    label_o: str = None,
    column_m: str = None,
    label_m: str = None,
    ylabel: str = None,
    ptile: int = None,
    vdiff: float = None,
    outname: str = "plot",
    domain_type: str = None,
    domain_name: str = None,
    fig_dict: dict = None,
    text_dict: dict = None,
    debug: bool = False,
):
    """Creates surface spatial bias plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Model/obs paired data to plot.
    df_reg : pandas.DataFrame, optional
        Model/obs paired regulatory data to plot.
    column_o : str
        Column label of observation variable to plot.
    label_o : str
        Name of observation variable to use in plot title.
    column_m : str
        Column label of model variable to plot.
    label_m : str
        Name of model variable to use in plot title.
    ylabel : str, optional
        Title of colorbar axis.
    ptile : int, optional
        Percentile calculation (0-100).
    vdiff : float, optional
        Min and max value to use on colorbar axis.
    outname : str
        File location and name of plot (do not include .png).
    domain_type : str, optional
        Domain type specified in input yaml file.
    domain_name : str, optional
        Domain name specified in input yaml file.
    fig_dict : dict, optional
        Dictionary containing information about figure.
    text_dict : dict, optional
        Dictionary containing information about text.
    debug : bool
        Whether to plot interactively (True) or not (False).

    Returns
    -------
    None
        Saves the generated plot to a file.
    """
    if debug is False:
        plt.ioff()

    def_map = dict(states=True, figsize=[10, 5])
    if fig_dict is not None:
        map_kwargs = {**def_map, **fig_dict}
    else:
        map_kwargs = def_map

    # If not specified use the PlateCarree projection
    if "crs" not in map_kwargs:
        map_kwargs["crs"] = ccrs.PlateCarree()

    # set default text size
    def_text = dict(fontsize=20)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text

    # set ylabel to column if not specified.
    if ylabel is None:
        ylabel = column_o

    if ptile is None:
        ylabel = "Mean " + ylabel
    else:
        ylabel = "{:02d}".format(ptile) + "th percentile " + ylabel

    if df_reg is not None:
        if ptile is None:
            df_mean = df_reg.groupby(["siteid"], as_index=False).mean(numeric_only=True)
        else:
            df_mean = df_reg.groupby(["siteid"], as_index=False).quantile(ptile / 100.0, numeric_only=True)
        col1, col2 = column_o + "_reg", column_m + "_reg"
    else:
        if ptile is None:
            df_mean = df.groupby(["siteid"], as_index=False).mean(numeric_only=True)
        else:
            df_mean = df.groupby(["siteid"], as_index=False).quantile(ptile / 100.0, numeric_only=True)
        col1, col2 = column_o, column_m

    p = monet_plots.SpatialBiasScatterPlot(
        data=df_mean,
        col1=col1,
        col2=col2,
        ax=plt.gca() if plt.get_fignums() else None,
        cmap="OrangeBlue",
        vmax=vdiff,
        **map_kwargs,
    )
    ax = p.plot(edgecolor="k", linewidth=0.8)

    if domain_type == "all":
        latmin = 25.0
        lonmin = -130.0
        latmax = 50.0
        lonmax = -60.0
        plt.title(
            domain_name + ": " + label_m + " - " + label_o,
            fontweight="bold",
            **text_kwargs,
        )
    elif domain_type == "epa_region" and domain_name is not None:
        latmin, lonmin, latmax, lonmax, acro = get_epa_bounds(index=None, acronym=domain_name)
        plt.title(
            "EPA Region " + domain_name + ": " + label_m + " - " + label_o,
            fontweight="bold",
            **text_kwargs,
        )
    else:
        latmin = math.floor(min(df.latitude))
        lonmin = math.floor(min(df.longitude))
        latmax = math.ceil(max(df.latitude))
        lonmax = math.ceil(max(df.longitude))
        plt.title(
            domain_name + ": " + label_m + " - " + label_o,
            fontweight="bold",
            **text_kwargs,
        )

    if "extent" not in map_kwargs:
        map_kwargs["extent"] = [lonmin, lonmax, latmin, latmax]
    ax.axes.set_extent(map_kwargs["extent"], crs=ccrs.PlateCarree())

    # Update colorbar
    f = plt.gcf()
    model_ax = f.get_axes()[0]
    cax = f.get_axes()[1]
    # get the position of the plot axis and use this to rescale nicely the color bar to the height of the plot.
    position_m = model_ax.get_position()
    position_c = cax.get_position()
    cax.set_position(
        [
            position_c.x0,
            position_m.y0,
            position_c.x1 - position_c.x0,
            (position_m.y1 - position_m.y0) * 1.1,
        ]
    )
    cax.set_ylabel(ylabel, fontweight="bold", **text_kwargs)
    cax.tick_params(
        labelsize=text_kwargs["fontsize"] * 0.8,
        length=10.0,
        width=2.0,
        grid_linewidth=2.0,
    )

    # plt.tight_layout(pad=0)
    savefig(outname + ".png", loc=4, logo_height=120)


####NEW function for adding 'altitude' variable as secondary y- axis (qzr++)
def add_yax2_altitude(
    ax: plt.Axes,
    pairdf: pd.DataFrame,
    altitude_yax2: dict,
    text_kwargs: dict,
    vmin_y2: float,
    vmax_y2: float,
) -> plt.Axes:
    """Creates secondary y-axis (altitude) for timeseries plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes from previous occurrences to overlay obs and model results.
    pairdf : pandas.DataFrame
        Model/obs paired data to plot.
    altitude_yax2 : dict
        Secondary y-axis (altitude) control options.
    text_kwargs : dict
        Dictionary containing information about text properties.
    vmin_y2 : float
        Minimum value for the secondary y-axis.
    vmax_y2 : float
        Maximum value for the secondary y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """
    ax2 = ax.twinx()

    # Fetch altitude parameters from altitude_yax2
    altitude_variable = altitude_yax2["altitude_variable"]
    altitude_ticks = altitude_yax2["altitude_ticks"]
    plot_kwargs_y2 = altitude_yax2.get("plot_kwargs_y2", {})
    ylabel2 = altitude_yax2.get("ylabel2", "Altitude")

    # Plot altitude
    ax2.plot(pairdf.index, pairdf[altitude_variable], **plot_kwargs_y2, label=ylabel2)

    # Set labels, ticks, and limits
    ax2.set_ylabel(
        ylabel2,
        fontweight="bold",
        fontsize=text_kwargs["fontsize"],
        color=plot_kwargs_y2.get("color", "g"),
    )
    ax2.tick_params(
        axis="y",
        labelcolor=plot_kwargs_y2.get("color", "g"),
        labelsize=text_kwargs["fontsize"] * 0.8,
    )
    ax2.set_ylim(vmin_y2, vmax_y2)
    ax2.set_xlim(ax.get_xlim())
    start_tick = max(0, vmin_y2 - altitude_ticks)
    ax2.yaxis.set_ticks(np.arange(start_tick, vmax_y2 + altitude_ticks + 1, altitude_ticks))

    # Extract the current legend and add a custom legend for the altitude line
    lines, labels = ax.get_legend_handles_labels()
    lines.append(ax2.get_lines()[0])
    labels.append(ylabel2)
    ax.legend(
        lines,
        labels,
        frameon=False,
        fontsize=text_kwargs["fontsize"],
        bbox_to_anchor=(1.15, 0.9),
        loc="center left",
    )

    return ax


###NEW curtain plot qzr++  (NEW CURTAIN model plot with model overlay, shared x-axis
def make_curtain_plot(
    time: np.ndarray,
    altitude: np.ndarray,
    model_data_2d: np.ndarray,
    obs_pressure: np.ndarray,
    pairdf: pd.DataFrame,
    mod_var: str,
    obs_var: str,
    grp_dict: dict,
    vmin: float = None,
    vmax: float = None,
    cmin: float = None,
    cmax: float = None,
    plot_dict: dict = None,
    outname: str = "plot",
    domain_type: str = None,
    domain_name: str = None,
    obs_label_config: dict = None,
    text_dict: dict = None,
    debug: bool = False,
):
    """
    Generates a curtain plot comparing model data with obs across altitude (Pressure) over time.

    Two Subplots: 1) model data contourf plot with model scatter overlay,
    2) another for observation data using scatter plot.
    Shared X-Axis: The time axis is shared between the two plots for better comparison.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time points.
    altitude : numpy.ndarray
        Array of altitude points.
    model_data_2d : numpy.ndarray
        2D array of cleaned and interpolated model data.
    obs_pressure : numpy.ndarray
        Array of pressure points corresponding to observations.
    pairdf : pandas.DataFrame
        Paired DataFrame containing model and observation data for scatter plots.
    mod_var : str
        Model variable name for labeling.
    obs_var : str
        Observation variable name for labeling.
    grp_dict : dict
        Plot configuration options.
    vmin : float, optional
        Min value to use on y-axis.
    vmax : float, optional
        Max value to use on y-axis.
    cmin : float, optional
        Minimum value for color normalization.
    cmax : float, optional
        Maximum value for color normalization.
    plot_dict : dict, optional
        Plotting information for each pair.
    outname : str
        File location and name of plot.
    domain_type : str, optional
        Type of domain being plotted.
    domain_name : str, optional
        Name of the domain being plotted.
    obs_label_config : dict, optional
        Configuration dictionary for observation labels.
    text_dict : dict, optional
        Dictionary containing text properties.
    debug : bool
        Whether to plot interactively (True) or not (False).

    Returns
    -------
    None
        Saves the generated plot to a file.
    """

    if model_data_2d.size == 0 or pairdf.empty:
        print("Warning: Model or observation data is empty. Skipping plot.")
        return

    # Debugging: print shapes and ndims
    print(f"time shape: {time.shape}, ndims: {time.ndim}")
    print(f"altitude shape: {altitude.shape}, ndims: {altitude.ndim}")
    print(f"model_data_2d shape: {model_data_2d.shape}, ndims: {model_data_2d.ndim}")
    print(f"pairdf shape: {pairdf.shape}, ndims: {pairdf.ndim}")

    # Determine cmin and cmax dynamically if not provided
    if cmin is None or cmax is None:
        cmin = min(np.nanmin(model_data_2d), np.nanmin(pairdf[obs_var].values))
        cmax = max(np.nanmax(model_data_2d), np.nanmax(pairdf[obs_var].values))

    print(f"cmin and cmax set dynamically based on model and observation data to {cmin} and {cmax}")

    fig, axs = plt.subplots(
        nrows=2,
        figsize=grp_dict.get("fig_kwargs", {}).get("figsize", (20, 8)),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Handling custom color maps
    if "color_map_custom" in grp_dict and grp_dict["color_map_custom"] is True and "colors" in grp_dict:
        # Custom color map logic
        colors = grp_dict["colors"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=grp_dict.get("color_levels", 100))
    else:
        # Default color map logic
        cmap = plt.get_cmap(grp_dict.get("color_map", "viridis"))

    norm = Normalize(vmin=cmin, vmax=cmax)

    time_dates = mdates.date2num(time)
    time_mesh, altitude_mesh = np.meshgrid(time_dates, altitude, indexing="ij")

    levels = np.linspace(cmin, cmax, 200)
    contourf_plot = axs[0].contourf(
        time_mesh,
        altitude_mesh,
        model_data_2d,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )

    # Adjust the layout to place the colorbar at the bottom
    cbar = fig.colorbar(
        contourf_plot,
        ax=axs.ravel().tolist(),
        orientation="horizontal",
        pad=0.1,
        aspect=50,
    )
    cbar.set_label(
        obs_label_config.get(obs_var, {}).get("ylabel_plot", "Variable (units)"),
        fontsize=text_dict.get("fontsize", 18),
        fontweight=text_dict.get("fontweight", "bold"),
    )
    cbar.ax.tick_params(labelsize=text_dict.get("labelsize", 14))

    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45, ha="right")

    # Set the main title and subplot titles
    fig.suptitle(
        f"Curtain Plot: {mod_var} vs {obs_var}",
        fontsize=text_dict.get("fontsize", 16),
        fontweight=text_dict.get("fontweight", "bold"),
    )
    axs[0].set_title(
        "Model Curtain with Model Scatter Overlay",
        fontsize=text_dict.get("fontsize", 18),
        fontweight=text_dict.get("fontweight", "bold"),
    )

    ##axs[0].set_ylabel('Pressure (Pa)', fontsize=text_dict.get('fontsize', 18), fontweight=text_dict.get('fontweight', 'bold')) #removed explicit y-axis label (made it flexible: see pressure_units via yaml)
    axs[0].invert_yaxis()  # Invert y-axis to have max pressure at the bottom
    axs[0].tick_params(axis="both", labelsize=text_dict.get("labelsize", 14))
    axs[1].invert_yaxis()  # Invert y-axis to have max pressure at the bottom (FOR SECOND SUBPLOT)

    # Set y-axis limits if vmin and vmax are provided
    if vmin is not None and vmax is not None:
        # Since y-axis is inverted already (see invert_yaxis() above), set limits accordingly
        print(f"Setting y-axis limits: vmin={vmin}, vmax={vmax}")
        axs[0].set_ylim(vmax, vmin)  # Note the order to match inversion
        axs[1].set_ylim(vmax, vmin)  # Note the order to match inversion
    else:
        vmin, vmax = axs[0].get_ylim()

    # Retrieve pressure units from grp_dict or default to 'Pa'
    pressure_units = grp_dict.get("pressure_units", "Pa")

    # Set y-tick labels and y-axis label based on pressure units
    if pressure_units == "hPa":
        y_axis_label = "Pressure (hPa)"
    else:
        y_axis_label = "Pressure (Pa)"

    # Apply y-tick labels and y-axis labels (both subplots)
    axs[0].set_ylabel(
        y_axis_label,
        fontsize=text_dict.get("fontsize", 18),
        fontweight=text_dict.get("fontweight", "bold"),
    )
    axs[1].set_ylabel(
        y_axis_label,
        fontsize=text_dict.get("fontsize", 18),
        fontweight=text_dict.get("fontweight", "bold"),
    )

    # Set y-axis ticks at specified intervals for axs[0] (first subplot)
    if "interval" in grp_dict:
        interval = grp_dict["interval"]
        print(f"Interval value: {interval}")  # This will print the interval value
        y_ticks = np.arange(vmin, vmax + interval, interval)
        y_ticks = y_ticks[y_ticks <= vmax]  # Ensure ticks do not exceed vmax
        print(f"Calculated y_ticks: {y_ticks}")
        axs[0].set_yticks(y_ticks)
        # Format y-tick labels
        axs[0].set_yticklabels([str(int(tick)) for tick in y_ticks])
    else:
        y_ticks = axs[0].get_yticks()

    # Synchronize y-ticks and y-limits for axs[1] (Second subplot) based on axs[0] (First subplot)
    y_ticks = axs[0].get_yticks()
    y_lim = axs[0].get_ylim()
    axs[1].set_yticks(y_ticks)
    axs[1].set_ylim(y_lim)
    # Set y-tick labels for axs[1]
    axs[1].set_yticklabels(
        [str(int(tick)) for tick in y_ticks]
    )  ##axs[1].set_yticklabels([str(int(tick)) if tick != y_ticks.min() else '' for tick in y_ticks]) #if needed to hide minimum

    # Use migrated monet-plots logic
    p = monet_plots.AircraftCurtainPlot(
        time=time,
        altitude=altitude,
        model_data_2d=model_data_2d,
        obs_pressure=obs_pressure,
        pairdf=pairdf,
        mod_var=mod_var,
        obs_var=obs_var,
        fig=fig,
        ax=axs[0],
    )
    p.plot(cmin=cmin, cmax=cmax, cmap=cmap, norm=norm)

    # Re-apply titles and labels from grp_dict/text_dict
    axs[1].set_title(
        "Observation Scatter",
        fontsize=text_dict.get("fontsize", 18),
        fontweight=text_dict.get("fontweight", "bold"),
    )

    # Save the curtain plot for the current pair immediately
    print(f"Saving curtain plot to {outname}...")
    savefig(f"{outname}", loc=4, logo_height=250, dpi=300)
    plt.show()

    if not debug:
        plt.close()

    # Diagnostic Histogram
    # plt.figure(figsize=(10, 4))
    # plt.hist(model_data_2d.values.flatten(), bins=100, alpha=0.15, color='blue', label='Model', range=(cmin, cmax))
    # plt.hist(obs_data_2d.flatten(), bins=100, alpha=0.5, color='green', label='Observation', range=(cmin, cmax))
    # plt.legend()
    # plt.title('Distribution of Model and Observation Data')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()


####NEW vertprofile has option for both shading (for interquartile range) or box (interquartile range)-whisker (10th-90th percentile bounds) (qzr++)
def make_vertprofile(
    df: pd.DataFrame,
    column: str = None,
    label: str = None,
    ax: plt.Axes = None,
    bins: int = None,
    altitude_variable: str = None,
    ylabel: str = None,
    vmin: float = None,
    vmax: float = None,
    domain_type: str = None,
    domain_name: str = None,
    plot_dict: dict = None,
    fig_dict: dict = None,
    text_dict: dict = None,
    debug: bool = False,
    interquartile_style: str = None,
) -> plt.Axes:
    """Creates altitude profile plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Model/obs paired data to plot.
    column : str, optional
        Column label of variable to plot.
    label : str, optional
        Name of variable to use in plot legend.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes from previous occurrence to overlay results.
    bins : int or array-like, optional
        Bins for binning the altitude variable.
    altitude_variable : str, optional
        The altitude variable in the paired df.
    ylabel : str, optional
        Title of y-axis.
    vmin : float, optional
        Min value to use on y-axis.
    vmax : float, optional
        Max value to use on y-axis.
    domain_type : str, optional
        Domain type specified in input yaml file.
    domain_name : str, optional
        Domain name specified in input yaml file.
    plot_dict : dict, optional
        Plotting information for each pair.
    fig_dict : dict, optional
        Dictionary containing information about the figure.
    text_dict : dict, optional
        Dictionary containing information about text.
    debug : bool
        Whether to plot interactively (True) or not (False).
    interquartile_style : str, optional
        Whether the vertical profile uses shading or box style for interquartile range.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """
    if debug is False:
        plt.ioff()

    # First, define items for all plots
    # Set default text size
    def_text = dict(fontsize=14)
    if text_dict is not None:
        text_kwargs = {**def_text, **text_dict}
    else:
        text_kwargs = def_text

    # Set ylabel to column if not specified
    if ylabel is None:
        ylabel = column
    if label is not None:
        plot_dict["label"] = label
    if vmin is not None and vmax is not None:
        plot_dict["ylim"] = [vmin, vmax]
    # Scale the fontsize for the x and y labels by the text_kwargs
    plot_dict["fontsize"] = text_kwargs["fontsize"] * 0.8

    # Then, if no plot has been created yet, create a plot and plot the obs
    if ax is None:
        # First define the colors for the observations
        obs_dict = dict(color="k", linestyle="-", marker="*", linewidth=1.2, markersize=6.0)
        if plot_dict is not None:
            # Whatever is not defined in the yaml file is filled in with the obs_dict here
            plot_kwargs = {**obs_dict, **plot_dict}
        else:
            plot_kwargs = obs_dict
        # Create the figure
        if fig_dict is not None:
            f, ax = plt.subplots(**fig_dict)
        else:
            f, ax = plt.subplots(figsize=(10, 6))

        # Bin the altitude variable and calculate median and interquartiles
        altitude_bins = pd.cut(df[altitude_variable], bins=bins)
        # Calculate the midpoints of the altitude bins
        bin_midpoints = altitude_bins.apply(lambda x: x.mid)
        # Convert bin_midpoints to a column in the DataFrame
        df["bin_midpoints"] = bin_midpoints
        median = df.groupby(altitude_bins, observed=True)[column].median(numeric_only=True)
        q1 = df.groupby(altitude_bins, observed=True)[column].quantile(0.25, numeric_only=True)
        q3 = df.groupby(altitude_bins, observed=True)[column].quantile(0.75, numeric_only=True)
        # Convert bin_midpoints to a numerical data type
        df["bin_midpoints"] = df["bin_midpoints"].astype(float)

        # p5 = df.groupby(altitude_bins, observed=True)[column].quantile(0.05, numeric_only=True)
        p10 = df.groupby(altitude_bins, observed=True)[column].quantile(0.10, numeric_only=True)
        p90 = df.groupby(altitude_bins, observed=True)[column].quantile(0.90, numeric_only=True)
        # p95 = df.groupby(altitude_bins, observed=True)[column].quantile(0.95, numeric_only=True)

        # Calculate the mean of bin_midpoints grouped by altitude bins
        binmidpoint = df.groupby(altitude_bins, observed=True)["bin_midpoints"].mean(numeric_only=True)

        ##Plotting vertprofile starts
        plot_kwargs_fillbetween = plot_kwargs.copy()
        del plot_kwargs_fillbetween["marker"]
        del plot_kwargs_fillbetween["markersize"]
        del plot_kwargs_fillbetween["fontsize"]

        # Copy the plot_kwargs outside the loop
        plot_kwargs_fillbox = plot_kwargs.copy()

        # Define box properties for transparency
        boxprops = {
            "facecolor": "none",  # Transparent facecolor
        }

        # Track labels that have been added to the legend
        # labels_added_to_legend = set()

        # Plot shaded interquartiles
        plot_kwargs_fillbetween["label"] = f"{label} (interquartile 25th-75th percentile range)"
        if interquartile_style == "shading":
            ax.fill_betweenx(
                binmidpoint.values,
                q1.values,
                q3.values,
                alpha=0.3,
                **plot_kwargs_fillbetween,
            )

        # For interquartile_style == 'box' case
        elif interquartile_style == "box":
            box_data = []
            for idx in range(len(q1)):
                # Create a dictionary for each box with the required statistics
                box_data.append(
                    {
                        "med": median.values[idx],
                        "q1": q1.values[idx],
                        "q3": q3.values[idx],
                        "whislo": p10.values[idx],
                        "whishi": p90.values[idx],
                        "fliers": [],
                    }
                )

            # Creating a vertical boxplot for each bin
            for idx, data in enumerate(box_data):
                # Get the color for the label from plot_kwargs or use a default color
                color = plot_kwargs_fillbox.get("color", "black")

                # Determine if the label should be added to the legend (only once for each label)
                legend_label = label if idx == 0 else None

                # Box properties
                boxprops = {
                    "facecolor": "none",  # Transparent face color
                    "edgecolor": color,  # Edge color matching the median line color
                    "linewidth": 2,  # Thickness of the box
                }

                # Whisker properties
                whiskerprops = {
                    "color": color,  # Whisker color
                    "linewidth": 2,  # Thickness of the whiskers
                }

                # Median line properties
                medianprops = {
                    "color": "none"  # Transparent color for median line
                }

                capprops = {
                    "color": color,
                    "linewidth": 2,
                }  # Whisker end bar properties

                # Create the box plot
                ax.bxp(
                    [data],
                    vert=False,
                    positions=[binmidpoint.values[idx]],
                    widths=500,
                    meanline=True,
                    patch_artist=True,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    medianprops=medianprops,
                    capprops=capprops,
                )

                # Optional: Add a legend entry for the box (only once)
                if legend_label and idx == 0:
                    box_legend_artist = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=color, linewidth=2)
                    ax.legend(
                        [box_legend_artist],
                        [f"{legend_label} (box-whisker)"],
                        loc="upper left",
                    )

        plot_kwargs["label"] = f"{label} (median)"
        median_line_df = pd.DataFrame(data={"median": median.values, "binmidpoint": binmidpoint.values})
        ax = median_line_df.plot(x="median", y="binmidpoint", ax=ax, legend=True, **plot_kwargs)

    # If plot has been created, add to the current axes
    else:
        # This means that an axis handle already exists, so use it to plot the model output
        altitude_bins = pd.cut(df[altitude_variable], bins=bins)
        # Calculate the midpoints of the altitude bins
        bin_midpoints = altitude_bins.apply(lambda x: x.mid)
        # Convert bin_midpoints to a column in the DataFrame
        df["bin_midpoints"] = bin_midpoints
        # can be .groupby(bin_midpoints) as well (qzr)
        median = df.groupby(altitude_bins, observed=True)[column].median(numeric_only=True)
        q1 = df.groupby(altitude_bins, observed=True)[column].quantile(0.25, numeric_only=True)
        q3 = df.groupby(altitude_bins, observed=True)[column].quantile(0.75, numeric_only=True)
        # Convert bin_midpoints to a numerical data type
        df["bin_midpoints"] = df["bin_midpoints"].astype(float)

        # Calculate the 10th, 90th, 5th, and 95th percentiles
        p10 = df.groupby(altitude_bins, observed=True)[column].quantile(0.10, numeric_only=True)
        p90 = df.groupby(altitude_bins, observed=True)[column].quantile(0.90, numeric_only=True)
        # p5 = df.groupby(altitude_bins, observed=True)[column].quantile(0.05, numeric_only=True)
        # p95 = df.groupby(altitude_bins, observed=True)[column].quantile(0.95, numeric_only=True)

        # Calculate the mean of bin_midpoints grouped by altitude bins
        binmidpoint = df.groupby(altitude_bins, observed=True)["bin_midpoints"].mean(numeric_only=True)

        plot_kwargs_fillbetween = plot_dict.copy()
        del plot_kwargs_fillbetween["marker"]
        del plot_kwargs_fillbetween["markersize"]
        del plot_kwargs_fillbetween["fontsize"]

        # Copy the plot_kwargs outside the loop
        plot_kwargs_fillbox = plot_dict.copy()

        # Define box properties for transparency
        boxprops = {
            "facecolor": "none",  # Transparent facecolor
        }

        # Track labels that have been added to the legend
        # labels_added_to_legend = set()

        plot_kwargs_fillbetween["label"] = f"{label} (interquartile 25th-75th percentile range)"
        if interquartile_style == "shading":
            ax.fill_betweenx(
                binmidpoint.values,
                q1.values,
                q3.values,
                alpha=0.3,
                **plot_kwargs_fillbetween,
            )
        # For interquartile_style == 'box' case
        elif interquartile_style == "box":
            box_data = []
            for idx in range(len(q1)):
                # Create a dictionary for each box with the required statistics
                box_data.append(
                    {
                        "med": median.values[idx],
                        "q1": q1.values[idx],
                        "q3": q3.values[idx],
                        "whislo": p10.values[idx],
                        "whishi": p90.values[idx],
                        "fliers": [],
                    }
                )

            # Creating a vertical boxplot for each bin
            for idx, data in enumerate(box_data):
                # Get the color for the label from plot_kwargs or use a default color
                color = plot_kwargs_fillbox.get("color", "black")

                # Determine if the label should be added to the legend (only once for each label)
                legend_label = label if idx == 0 else None

                # Box properties
                boxprops = {
                    "facecolor": "none",  # Transparent face color
                    "edgecolor": color,  # Edge color matching the median line color
                    "linewidth": 2,  # Thickness of the box
                }

                # Whisker properties
                whiskerprops = {
                    "color": color,  # Whisker color
                    "linewidth": 2,  # Thickness of the whiskers
                }

                # Median line properties
                medianprops = {
                    "color": "none"  # Transparent color for median line
                }

                capprops = {
                    "color": color,
                    "linewidth": 2,
                }  # Whisker end bar properties

                # Create the box plot
                ax.bxp(
                    [data],
                    vert=False,
                    positions=[binmidpoint.values[idx]],
                    widths=500,
                    meanline=True,
                    patch_artist=True,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    medianprops=medianprops,
                    capprops=capprops,
                )

                # Optional: Add a legend entry for the box (only once)
                if legend_label and idx == 0:
                    box_legend_artist = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=color, linewidth=2)
                    ax.legend(
                        [box_legend_artist],
                        [f"{legend_label} (box-whisker)"],
                        loc="upper left",
                    )

        # Plot median line
        plot_dict["label"] = f"{label} (median)"
        median_line_df = pd.DataFrame(data={"median": median.values, "binmidpoint": binmidpoint.values})
        ax = median_line_df.plot(x="median", y="binmidpoint", ax=ax, legend=True, **plot_dict)

    if interquartile_style == "box":
        # Add text to legend (adjust the x and y coordinates to place the text below the legend)
        plt.text(
            1.12,
            0.7,
            "Bounds of box: Interquartile range\nWhiskers: 10th and 90th percentiles",
            transform=ax.transAxes,
            fontsize=text_kwargs["fontsize"] * 0.8,
        )
    # Apply the custom formatter to the y-axis (round off y-axis tick labels if after decimal its just zero)
    ax.yaxis.set_major_formatter(FuncFormatter(custom_yaxis_formatter))
    # Set parameters for all plots
    ax.set_ylabel("Altitude (m)", fontweight="bold", **text_kwargs)
    ax.set_xlabel(ylabel, fontweight="bold", **text_kwargs)
    ax.legend(frameon=False, fontsize=text_kwargs["fontsize"] * 0.8)
    ax.tick_params(axis="both", length=10.0, direction="inout")
    ax.tick_params(axis="both", which="minor", length=5.0, direction="out")
    # Adjust label position
    ax.legend(
        frameon=False,
        fontsize=text_kwargs["fontsize"] * 0.8,
        bbox_to_anchor=(1.1, 0.9),
        loc="center left",
    )

    if domain_type is not None and domain_name is not None:
        if domain_type == "epa_region":
            ax.set_title("EPA Region " + domain_name, fontweight="bold", **text_kwargs)
        else:
            ax.set_title(domain_name, fontweight="bold", **text_kwargs)

    return ax


##NEW Scatter Density Plot for model obs pairs (matplotlib scatter plot if fill=False or seaborn kde sactter density plot if fill= True)
def make_scatter_density_plot(
    df: pd.DataFrame,
    mod_var: str = None,
    obs_var: str = None,
    ax: plt.Axes = None,
    color_map: str = "viridis",
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    fill: bool = False,
    vmin_x: float = None,
    vmax_x: float = None,
    vmin_y: float = None,
    vmax_y: float = None,
    outname: str = "plot",
    **kwargs,
) -> plt.Axes:
    """
    Creates a scatter density plot for model/obs pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Paired DataFrame containing the model and observation data.
    mod_var : str, optional
        Model variable name.
    obs_var : str, optional
        Observation variable name.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes from a previous occurrence.
    color_map : str, optional
        Colormap for the density.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Title for the plot.
    fill : bool
        If True, creates a Seaborn KDE plot. If False, a scatter plot.
    vmin_x : float, optional
        Min value for x-axis.
    vmax_x : float, optional
        Max value for x-axis.
    vmin_y : float, optional
        Min value for y-axis.
    vmax_y : float, optional
        Max value for y-axis.
    outname : str
        File location and name of plot.
    **kwargs : dict
        Additional keyword arguments for customization.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """

    # Create a custom colormap based on color_map options in yaml or just use default colormap id color_map is just a string (e.g. viridis)
    # Determine the normalization based on vcenter
    vcenter = kwargs.get("vcenter", None)

    if vcenter is not None:
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin_x, vmax=vmax_x)
    else:
        norm = None  # This means we'll use a default linear normalization

    extensions = kwargs.get("extensions", None)  # Extract extensions for the colorbar

    # Check if the color_map key from the YAML file provides a dictionary
    # (indicating a custom colormap) or just a string (indicating a built-in colormap like 'magma', 'viridis' etc.).
    color_map_config = color_map

    # print(f"Color Map Config: {color_map_config}") #Debugging

    if isinstance(color_map_config, dict):
        colors = color_map_config["colors"]
        over = color_map_config.get("over", None)
        under = color_map_config.get("under", None)

        cmap = mpl.colors.ListedColormap(colors).with_extremes(over=over, under=under)
    else:
        cmap = plt.get_cmap(color_map_config)

    # Debug print statement to check the colormap configuration
    # print(f"Using colormap: {cmap}") #Debugging

    if isinstance(cmap, mpl.colors.ListedColormap):
        cmap = LinearSegmentedColormap.from_list("custom", cmap.colors)

    # Check if 'ax' is None and create a new subplot if needed
    if ax is None:
        fig, ax = plt.subplots()

    x_data = df[mod_var]
    y_data = df[obs_var]

    if fill:  # For KDE plot
        # print("Generating KDE plot...")

        # Check the type of the colormap and set Seaborn's palette accordingly
        if isinstance(cmap, mpl.colors.ListedColormap):
            sns.set_palette(cmap.colors)
        elif isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            # If it's a LinearSegmentedColormap, extract N colors from the colormap
            N = 256
            sns.set_palette([cmap(i) for i in range(N)])

        # Create the KDE fill plot using seaborn
        plot = sns.kdeplot(
            x=x_data.dropna(),
            y=y_data.dropna(),
            cmap=cmap,
            norm=norm,
            fill=True,
            ax=ax,
            **{k: v for k, v in kwargs.items() if k in sns.kdeplot.__code__.co_varnames},
        )
        colorbar_label = "Density"

        # Get the QuadMesh object from the Axes for the colorbar and explicitly set its colormap
        mappable = ax.collections[0]
        mappable.set_cmap(cmap)

    else:  # For scatter plot using matplotlib
        # print("Generating scatter plot...")
        plot = plt.scatter(
            x_data,
            y_data,
            c=y_data,
            cmap=cmap,
            norm=norm,
            marker="o",
            **{k: v for k, v in kwargs.items() if k in plt.scatter.__code__.co_varnames},
        )
        units = ylabel[ylabel.find("(") + 1 : ylabel.find(")")]
        colorbar_label = units  # Units for scatter plot
        mappable = plot

    # Set plot labels and titles
    if xlabel:
        plt.xlabel(xlabel, fontweight="bold")
    if ylabel:
        plt.ylabel(ylabel, fontweight="bold")
    if title:
        plt.title(title, fontweight="bold")
    if vmin_x is not None:
        plt.xlim(left=vmin_x)
    if vmax_x is not None:
        plt.xlim(right=vmax_x)
    if vmin_y is not None:
        plt.ylim(bottom=vmin_y)
    if vmax_y is not None:
        plt.ylim(top=vmax_y)

    # Handle the colorbar using the mappable object
    if extensions:
        cbar = plt.colorbar(mappable, extend="both", ax=ax)  # Extends the colorbar at both ends
    else:
        cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label(colorbar_label)

    # Save the scatter density plot for the current pair immediately
    print(f"Saving scatter density plot to {outname}...")
    savefig(f"{outname}", loc=4, logo_height=100, dpi=300)
    plt.show()

    return ax


##NEW Violin plot
def calculate_violin(
    df: pd.DataFrame,
    column: str = None,
    label: str = None,
    plot_dict: dict = None,
    comb_violin: pd.DataFrame = None,
    label_violin: list = None,
) -> tuple:
    """
    Combines data into an acceptable format for violin plots.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the model/obs paired data to plot.
    column : str, optional
        Column label of the variable to plot.
    label : str, optional
        Name of the variable to use in the plot legend.
    plot_dict : dict, optional
        Dictionary containing color information for the plot.
    comb_violin : pandas.DataFrame, optional
        DataFrame containing information to create violin plots from previous occurrences.
    label_violin : list, optional
        List of dictionaries with labels and colors.

    Returns
    -------
    tuple
        (comb_violin, label_violin)
    """
    if comb_violin is None and label_violin is None:
        comb_violin = pd.DataFrame()
        label_violin = []
    # First define the colors for the observations.
    obs_dict = dict(color="gray", linestyle="-", marker="x", linewidth=1.2, markersize=6.0)
    if plot_dict is not None:
        # Whatever is not defined in the yaml file is filled in with the obs_dict here.
        plot_kwargs = {**obs_dict, **plot_dict}
    ##else:
    ##  plot_kwargs = obs_dict
    # else:
    ##plot_kwargs = plot_dict
    # For all, a column to the dataframe and append the label info to the list.
    plot_kwargs["column"] = column
    plot_kwargs["label"] = label
    ##if df_reg is not None:
    ##  comb_violin[label] = df_reg[column+'_reg']
    ##else:
    comb_violin[label] = df[column]
    label_violin.append(plot_kwargs)

    return comb_violin, label_violin


def make_violin_plot(
    comb_violin: pd.DataFrame,
    label_violin: list,
    outname: str = "plot",
    domain_type: str = None,
    domain_name: str = None,
    fig_dict: dict = None,
    text_dict: dict = None,
    debug: bool = False,
    ylabel: str = None,
    vmin: float = None,
    vmax: float = None,
):
    """
    Creates a violin plot from combined datasets.

    Parameters
    ----------
    comb_violin : pandas.DataFrame
        DataFrame containing combined data for all datasets.
    label_violin : list
        List of dictionaries with labels and colors.
    outname : str
        File location and name of plot (do not include .png).
    domain_type : str, optional
        Domain type specified in the input yaml file.
    domain_name : str, optional
        Domain name specified in the input yaml file.
    fig_dict : dict, optional
        Dictionary containing figure properties.
    text_dict : dict, optional
        Dictionary containing text properties.
    debug : bool
        If True, the plot will be shown interactively.
    ylabel : str, optional
        The label for the y-axis.
    vmin : float, optional
        The minimum value for the y-axis.
    vmax : float, optional
        The maximum value for the y-axis.

    Returns
    -------
    None
        Saves the generated plot to a file.
    """

    if not debug:
        plt.ioff()

    # Set the order and palette based on label_violin
    order = [item["label"] for item in label_violin]
    palette = {item["label"]: item["color"] for item in label_violin}

    # Initialize figure
    if fig_dict is not None:
        plt.figure(**fig_dict)

    # Convert the DataFrame to long-form or "tidy" format suitable for sns.violinplot
    melted_comb_violin = pd.melt(comb_violin, var_name="group", value_name="value")

    # Set labels and title using text_dict
    # Set default text size, modify here for bigger text
    def_text = dict(fontsize=14)  # can increase fontsize for default text (> 14)
    text_kwargs = {**def_text, **text_dict} if text_dict else def_text

    # Use migrated monet-plots logic
    p = monet_plots.ViolinPlot(
        data=melted_comb_violin,
        x="group",
        y="value",
        hue="group",
        order=order,
        palette=palette,
        ax=plt.gca(),
    )
    p.plot()

    # Set labels and title with increased size
    plt.xlabel("", weight="bold", fontsize=text_kwargs["fontsize"])
    plt.ylabel(ylabel if ylabel else "Value", weight="bold", fontsize=text_kwargs["fontsize"])

    # Increase tick label size
    plt.tick_params(axis="both", labelsize=text_kwargs["fontsize"] * 0.8)

    # Set y-axis limits if provided
    if vmin is not None and vmax is not None:
        plt.ylim(vmin, vmax)

    # Set title
    if domain_type and domain_name:
        plt.title(
            f"Violin Plot for {domain_type} - {domain_name}",
            fontweight="bold",
            fontsize=text_kwargs["fontsize"],
        )

    # Finalize and save plot
    plt.tight_layout()
    savefig(f"{outname}.png", loc=3, logo_height=200, dpi=300)

    # Close the plot if not in debug mode
    if not debug:
        plt.close()
