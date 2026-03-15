# SPDX-License-Identifier: Apache-2.0
#
import inspect
import os

import monet_plots
import pandas as pd
import xarray as xr

from melodies_monet.stats.proc_stats import calc as mm_calc


def compute_stats(paired_dict, output_dir="./", debug=False, **kwargs):
    """
    Bridge function to compute statistics using monet-stats.

    Parameters
    ----------
    paired_dict : dict
        Dictionary of pair objects.
    output_dir : str
        Directory to save results.
    debug : bool
        Enable debug output.
    **kwargs : Any
        Additional arguments from the YAML.
    """
    results = {}
    stat_list = kwargs.get("stat_list", [])

    if not stat_list:
        stat_list = ["MB", "MAE", "RMSE", "R", "IOA"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for label, p_inst in paired_dict.items():
        if debug:
            print(f"Computing stats for {label}")

        from melodies_monet.stats.proc_stats import create_table, produce_stat_dict

        stat_full_names = produce_stat_dict(stat_list)

        columns = p_inst.model_vars
        stat_df = pd.DataFrame(index=stat_list, columns=columns)
        stat_df["Stat_FullName"] = stat_full_names

        for stat in stat_list:
            for i, modvar in enumerate(p_inst.model_vars):
                obsvar = p_inst.ref_vars[i]
                # Pass all kwargs to calc, which will pass them to monet_stats functions
                val = mm_calc(p_inst.obj, stat=stat, obsvar=obsvar, modvar=modvar, **kwargs)
                stat_df.loc[stat, modvar] = val

        # Save to CSV - Need to compute scalar values for CSV writing
        compute_df = stat_df.copy()
        for col in p_inst.model_vars:
            for idx in stat_list:
                v = compute_df.loc[idx, col]
                if hasattr(v, "compute"):
                    res = v.compute()
                    compute_df.loc[idx, col] = res.item() if hasattr(res, "item") else res

        outname = os.path.join(output_dir, f"stats_{label}")
        compute_df.to_csv(outname + ".csv")

        # Create table plot if requested
        if kwargs.get("create_table", True):
            create_table(
                stat_df, outname=outname, title=f"Stats for {label}", out_table_kwargs=kwargs.get("out_table_kwargs"), debug=debug
            )

        results[label] = stat_df

    return results


def create_plots(paired_dict, output_dir="./", debug=False, **kwargs):
    """
    Truly generic bridge function to generate plots using monet-plots.

    Parameters
    ----------
    paired_dict : dict
        Dictionary of pair objects.
    output_dir : str
        Directory to save results.
    debug : bool
        Enable debug output.
    **kwargs : Any
        Additional arguments from the YAML.
    """
    plot_type = kwargs.get("type", "timeseries").lower()

    # Discovery: Find the plot class in monet_plots
    plot_class = None

    for name, obj in inspect.getmembers(monet_plots):
        if inspect.isclass(obj) and (name.lower() == plot_type.lower() or name.lower() == plot_type.lower() + "plot"):
            plot_class = obj
            break

    if plot_class is None:
        print(f"Warning: Plot type '{plot_type}' not found in monet-plots.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for label, p_inst in paired_dict.items():
        if debug:
            print(f"Generating {plot_type} plot for {label}")

        # Global plotting configuration
        group_kwargs = kwargs.copy()
        for k in ["type", "data", "output_dir", "debug", "add_logo"]:
            group_kwargs.pop(k, None)

        group_plot_kwargs = group_kwargs.pop("plot_kwargs", {})

        for i, modvar in enumerate(p_inst.model_vars):
            obsvar = p_inst.ref_vars[i]

            data_to_plot = p_inst.obj

            # Identification of dimensions and coordinates
            time_col = None
            if hasattr(data_to_plot, "coords"):
                for c in ["time", "datetime", "date", "time_local"]:
                    if c in data_to_plot.coords or c in data_to_plot.dims:
                        time_col = c
                        break

            # Subplot keyword arguments handling
            subplot_keys = ["figsize", "nrows", "ncols", "sharex", "sharey", "subplot_kw", "gridspec_kw"]

            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(**{k: v for k, v in group_kwargs.items() if k in subplot_keys})

                sig_init = inspect.signature(plot_class.__init__)

                if plot_type == "timeseries" and isinstance(data_to_plot, xr.Dataset):
                    # Multi-line plot on same ax
                    y_vars = [obsvar, modvar]
                    for j, y_var in enumerate(y_vars):
                        current_kwargs = p_inst.ref_plot_kwargs if j == 0 else p_inst.model_plot_kwargs
                        current_combined = {**current_kwargs, **group_plot_kwargs}

                        init_args = {"data": data_to_plot, "df": data_to_plot, "x": time_col, "y": y_var, "ax": ax, "fig": fig}
                        final_init_args = {k: v for k, v in init_args.items() if k in sig_init.parameters}
                        p = plot_class(**final_init_args)

                        sig_plot = inspect.signature(p.plot)
                        final_plot_args = {k: v for k, v in current_combined.items() if k in sig_plot.parameters}
                        if sig_plot.parameters.get("kwargs") is not None:
                            final_plot_args.update({k: v for k, v in current_combined.items() if k not in final_plot_args})

                        p.plot(**final_plot_args)
                else:
                    # Generic handling
                    init_args = {
                        "data": data_to_plot,
                        "df": data_to_plot,
                        "x": time_col,
                        "y": [obsvar, modvar],
                        "ax": ax,
                        "fig": fig,
                    }
                    if "col1" in sig_init.parameters:
                        init_args["col1"] = obsvar
                    if "col2" in sig_init.parameters:
                        init_args["col2"] = modvar

                    final_init_args = {k: v for k, v in init_args.items() if k in sig_init.parameters}
                    p = plot_class(**final_init_args)

                    sig_plot = inspect.signature(p.plot)
                    # Use combined_plot_kwargs (pair-specific + group-level)
                    combined_plot_kwargs = {**p_inst.model_plot_kwargs, **group_plot_kwargs}
                    final_plot_args = {k: v for k, v in combined_plot_kwargs.items() if k in sig_plot.parameters}
                    if sig_plot.parameters.get("kwargs") is not None:
                        final_plot_args.update({k: v for k, v in combined_plot_kwargs.items() if k not in final_plot_args})

                    p.plot(**final_plot_args)

                if kwargs.get("add_logo", True) and hasattr(p, "add_logo"):
                    p.add_logo()

                outname = os.path.join(output_dir, f"{plot_type}_{label}_{modvar}.png")
                p.save(outname)
                p.close()
            except Exception as e:
                print(f"Error generating plot {label} for variable {modvar}: {e}")
                if debug:
                    import traceback

                    traceback.print_exc()
