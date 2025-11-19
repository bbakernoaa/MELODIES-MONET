# SPDX-License-Identifier: Apache-2.0
#
"""
The plotting module for MELODIES-MONET.
"""
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def plotting(analysis):
    """Cycle through all the plotting groups (e.g., plot_grp1) listed in
    the input yaml file and create the plots.

    This routine loops over all the domains and
    model/obs pairs specified in the plotting group (``.control_dict['plots']``)
    for all the variables specified in the mapping dictionary listed in
    :attr:`paired`.

    Creates plots stored in the file location specified by output_dir
    in the analysis section of the yaml file.

    Parameters
    ----------
    analysis : melodies_monet.driver.analysis
        The analysis object.

    Returns
    -------
    None
    """

    from ..util.tools import resample_stratify
    from ..util.region_select import select_region
    pair_keys = list(analysis.paired.keys())
    if analysis.paired[pair_keys[0]].type.lower() in ['sat_grid_clm','sat_swath_clm']:
        from ..plots import satplots as splots,savefig
    else:
        from ..plots import surfplots as splots, savefig
        from ..plots import aircraftplots as airplots
        from ..plots import sonde_plots as sondeplots
    from ..plots import xarray_plots as xrplots
    if not analysis.add_logo:
        savefig.keywords.update(decorate=False)

    # Disable figure count warning
    initial_max_fig = plt.rcParams["figure.max_open_warning"]
    plt.rcParams["figure.max_open_warning"] = 0

    # first get the plotting dictionary from the yaml file
    plot_dict = analysis.control_dict['plots']
    # Calculate any items that do not need to recalculate each loop.
    startdatename = str(datetime.datetime.strftime(analysis.start_time, '%Y-%m-%d_%H'))
    enddatename = str(datetime.datetime.strftime(analysis.end_time, '%Y-%m-%d_%H'))
    # now we are going to loop through each plot_group (note we can have multiple plot groups)
    # a plot group can have
    #     1) a singular plot type
    #     2) multiple paired datasets or model datasets depending on the plot type
    #     3) kwargs for creating the figure ie size and marker (note the default for obs is 'x')

    # Loop through the plot_dict items
    for grp, grp_dict in plot_dict.items():

        # Read the interquartile_style argument (for vertprofile plot type) if it exists
        if grp_dict.get('type') == 'vertprofile':
            interquartile_style = grp_dict.get('data_proc', {}).get('interquartile_style', 'shading')
        else:
            interquartile_style = None

        pair_labels = grp_dict['data']
        # Get the plot type
        plot_type = grp_dict['type']

        #read-in special settings for multi-boxplot
        if plot_type == 'multi_boxplot':
            region_name = grp_dict['region_name']
            region_list = grp_dict['region_list']
            model_name_list = grp_dict['model_name_list']

        #read-in special settings for ozone sonde related plots
        if plot_type in {'vertical_single_date', 'vertical_boxplot_os', 'density_scatter_plot_os'}:
            altitude_range = grp_dict['altitude_range']
            altitude_method = grp_dict['altitude_method']
            station_name = grp_dict['station_name']
            monet_logo_position = grp_dict['monet_logo_position']
            cds = grp_dict['compare_date_single']
            release_time= datetime.datetime(cds[0],cds[1],cds[2],cds[3],cds[4],cds[5])

            if plot_type == 'vertical_boxplot_os':
                altitude_threshold_list = grp_dict['altitude_threshold_list']
            elif plot_type == 'density_scatter_plot_os':
                cmap_method = grp_dict['cmap_method']
                model_name_list = grp_dict['model_name_list']

        #read-in special settings for scorecard
        if plot_type == 'scorecard':
            region_list = grp_dict['region_list']
            region_name = grp_dict['region_name']
            urban_rural_name = grp_dict['urban_rural_name']
            urban_rural_differentiate_value = grp_dict['urban_rural_differentiate_value']
            better_or_worse_method = grp_dict['better_or_worse_method']
            model_name_list = grp_dict['model_name_list']

        #read-in special settings for csi plot
        if plot_type == 'csi':
            threshold_list = grp_dict['threshold_list']
            score_name = grp_dict['score_name']
            model_name_list = grp_dict['model_name_list']
            threshold_tick_style = grp_dict.get('threshold_tick_style',None)

        # first get the observational obs labels

        obs_vars = []
        for pair_label in pair_labels:
            obs_vars.extend(analysis.paired[pair_label].obs_vars)
        # Guarantee uniqueness of obs_vars, without altering order
        obs_vars = list(dict.fromkeys(obs_vars))

        # loop through obs variables
        for obsvar in obs_vars:
            # Loop also over the domain types. So can easily create several overview and zoomed in plots.
            domain_types = grp_dict.get('domain_type', [None])
            domain_names = grp_dict.get('domain_name', [None])
            domain_infos = grp_dict.get('domain_info', {})
            # Use only pair_labels containing obs_var
            pair_labels_obsvar = [p for p in pair_labels if obsvar in analysis.paired[p].obs_vars]
            for domain in range(len(domain_types)):
                domain_type = domain_types[domain]
                domain_name = domain_names[domain]
                domain_info = domain_infos.get(domain_name, None)
                for p_index, p_label in enumerate(pair_labels_obsvar):
                    p = analysis.paired[p_label]
                    obs_type = p.type

                    # find the pair model label that matches the obs var
                    index = p.obs_vars.index(obsvar)
                    modvar = p.model_vars[index]

                    # Adjust the modvar as done in pairing script, if the species name in obs and model are the same.
                    if obsvar == modvar:
                        modvar = modvar + '_new'

                    # Adjust the modvar for satellite no2 trop. column paring. M.Li
                    if obsvar == 'nitrogendioxide_tropospheric_column':
                        modvar = modvar + 'trpcol'

                    # for pt_sfc data, convert to pandas dataframe, format, and trim
                    # Query selected points if applicable
                    if domain_type != 'all':
                        p_region = select_region(p.obj, domain_type, domain_name, domain_info)
                    else:
                        p_region = p.obj


                    if obs_type in ["sat_swath_sfc", "sat_swath_clm", "sat_grid_sfc",
                                    "sat_grid_clm", "sat_swath_prof"]:
                         # convert index to time; setup for sat_swath_clm

                        if 'time' not in p_region.dims and obs_type == 'sat_swath_clm':
                            pairdf_all = p_region.swap_dims({'x':'time'})

                        else:
                            pairdf_all = p_region
                        # Select only the analysis time window.
                        pairdf_all = pairdf_all.sel(time=slice(analysis.start_time,analysis.end_time))
                    else:
                        # convert to dataframe
                        pairdf_all = p_region.to_dataframe(dim_order=["time", "x"])
                        # Select only the analysis time window.
                        pairdf_all = pairdf_all.loc[analysis.start_time : analysis.end_time]

                    # Determine the default plotting colors.
                    if 'default_plot_kwargs' in grp_dict.keys():
                        if analysis.models[p.model].plot_kwargs is not None:
                            plot_dict = {**grp_dict['default_plot_kwargs'], **analysis.models[p.model].plot_kwargs}
                        else:
                            plot_dict = {**grp_dict['default_plot_kwargs'], **splots.calc_default_colors(p_index)}
                        obs_dict = grp_dict['default_plot_kwargs']
                    else:
                        if analysis.models[p.model].plot_kwargs is not None:
                            plot_dict = analysis.models[p.model].plot_kwargs.copy()
                        else:
                            plot_dict = splots.calc_default_colors(p_index).copy()
                        obs_dict = None

                    # Determine figure_kwargs and text_kwargs
                    if 'fig_kwargs' in grp_dict.keys():
                        fig_dict = grp_dict['fig_kwargs']
                    else:
                        fig_dict = None
                    if 'text_kwargs' in grp_dict.keys():
                        text_dict = grp_dict['text_kwargs']
                    else:
                        text_dict = None

                    # Read in some plotting specifications stored with observations.
                    if p.obs in analysis.obs and analysis.obs[p.obs].variable_dict is not None:
                        if obsvar in analysis.obs[p.obs].variable_dict.keys():
                            obs_plot_dict = analysis.obs[p.obs].variable_dict[obsvar].copy()
                        else:
                            obs_plot_dict = {}
                    else:
                        obs_plot_dict = {}

                    # Specify ylabel if noted in yaml file.
                    if 'ylabel_plot' in obs_plot_dict.keys():
                        use_ylabel = obs_plot_dict['ylabel_plot']
                    else:
                        use_ylabel = None

                    # Determine if set axis values or use defaults
                    if grp_dict['data_proc'].get('set_axis', False):
                        if obs_plot_dict:  # Is not null
                            set_yaxis = True
                        else:
                            print('Warning: variables dict for ' + obsvar + ' not provided, so defaults used')
                            set_yaxis = False
                    else:
                        set_yaxis = False

                    # Determine to calculate mean values or percentile
                    if 'percentile_opt' in obs_plot_dict.keys():
                        use_percentile = obs_plot_dict['percentile_opt']
                    else:
                        use_percentile = None



                    # Determine outname
                    outname = "{}.{}.{}.{}.{}.{}.{}".format(grp, plot_type, obsvar, startdatename, enddatename, domain_type, domain_name)

                    # Query with filter options
                    if 'filter_dict' in grp_dict['data_proc'] and 'filter_string' in grp_dict['data_proc']:
                        raise Exception("""For plot group: {}, only one of filter_dict and filter_string can be specified.""".format(grp))
                    elif 'filter_dict' in grp_dict['data_proc']:
                        filter_dict = grp_dict['data_proc']['filter_dict']
                        for column in filter_dict.keys():
                            filter_vals = filter_dict[column]['value']
                            filter_op = filter_dict[column]['oper']
                            if filter_op == 'isin':
                                pairdf_all.query(f'{column} == {filter_vals}', inplace=True)
                            elif filter_op == 'isnotin':
                                pairdf_all.query(f'{column} != {filter_vals}', inplace=True)
                            else:
                                pairdf_all.query(f'{column} {filter_op} {filter_vals}', inplace=True)
                    elif 'filter_string' in grp_dict['data_proc']:
                        pairdf_all.query(grp_dict['data_proc']['filter_string'], inplace=True)

                    # Drop sites with greater than X percent NAN values
                    if 'rem_obs_by_nan_pct' in grp_dict['data_proc']:
                        grp_var = grp_dict['data_proc']['rem_obs_by_nan_pct']['group_var']
                        pct_cutoff = grp_dict['data_proc']['rem_obs_by_nan_pct']['pct_cutoff']

                        if grp_dict['data_proc']['rem_obs_by_nan_pct']['times'] == 'hourly':
                            # Select only hours at the hour
                            hourly_pairdf_all = pairdf_all.reset_index().loc[pairdf_all.reset_index()['time'].dt.minute==0,:]

                            # calculate total obs count, obs count with nan removed, and nan percent for each group
                            grp_fullcount = hourly_pairdf_all[[grp_var,obsvar]].groupby(grp_var).size().rename({0:obsvar})
                            grp_nonan_count = hourly_pairdf_all[[grp_var,obsvar]].groupby(grp_var).count() # counts only non NA values
                        else:
                            # calculate total obs count, obs count with nan removed, and nan percent for each group
                            grp_fullcount = pairdf_all[[grp_var,obsvar]].groupby(grp_var).size().rename({0:obsvar})
                            grp_nonan_count = pairdf_all[[grp_var,obsvar]].groupby(grp_var).count() # counts only non NA values

                        grp_pct_nan = 100 - grp_nonan_count.div(grp_fullcount,axis=0)*100

                        # make list of sites meeting condition and select paired data by this by this
                        grp_select = grp_pct_nan.query(obsvar + ' < ' + str(pct_cutoff)).reset_index()
                        pairdf_all = pairdf_all.loc[pairdf_all[grp_var].isin(grp_select[grp_var].values)]

                    # Drop NaNs if using pandas
                    if obs_type in ['pt_sfc','aircraft','mobile','ground','sonde']:
                        if grp_dict['data_proc']['rem_obs_nan'] is True:
                            # I removed drop=True in reset_index in order to keep 'time' as a column.
                            pairdf = pairdf_all.reset_index().dropna(subset=[modvar, obsvar])
                        else:
                            pairdf = pairdf_all.reset_index().dropna(subset=[modvar])
                    elif obs_type in ["sat_swath_sfc", "sat_swath_clm",
                                      "sat_grid_sfc", "sat_grid_clm",
                                      "sat_swath_prof"]:
                        # xarray doesn't need nan drop because its math operations seem to ignore nans
                        # MEB (10/9/24): Add statement to ensure model and obs variables have nans at the same place
                        pairdf = pairdf_all.where(pairdf_all[obsvar].notnull() & pairdf_all[modvar].notnull())

                    else:
                        print('Warning: set rem_obs_nan = True for regulatory metrics')
                        pairdf = pairdf_all.reset_index().dropna(subset=[modvar])

                    # JianHe: do we need provide a warning if pairdf is empty (no valid obsdata) for specific subdomain?
                    # MEB: pairdf.empty fails for data left in xarray format. isnull format works.
                    if pairdf[obsvar].isnull().all():
                        print('Warning: no valid obs found for '+domain_name)
                        continue

                    # JianHe: Determine if calculate regulatory values
                    cal_reg = obs_plot_dict.get('regulatory', False)

                    if cal_reg:
                        # Reset use_ylabel for regulatory calculations
                        if 'ylabel_reg_plot' in obs_plot_dict.keys():
                            use_ylabel = obs_plot_dict['ylabel_reg_plot']
                        else:
                            use_ylabel = None

                        df2 = (
                            pairdf.copy()
                            .groupby("siteid")
                            .resample('h', on='time_local')
                            .mean(numeric_only=True)
                            .reset_index()
                        )

                        if obsvar == 'PM2.5':
                            pairdf_reg = splots.make_24hr_regulatory(df2,[obsvar,modvar]).rename(index=str,columns={obsvar+'_y':obsvar+'_reg',modvar+'_y':modvar+'_reg'})
                        elif obsvar == 'OZONE':
                            pairdf_reg = splots.make_8hr_regulatory(df2,[obsvar,modvar]).rename(index=str,columns={obsvar+'_y':obsvar+'_reg',modvar+'_y':modvar+'_reg'})
                        else:
                            print('Warning: no regulatory calculations found for ' + obsvar + '. Skipping plot.')
                            del df2
                            continue
                        del df2
                        if len(pairdf_reg[obsvar+'_reg']) == 0:
                            print('No valid data for '+obsvar+'_reg. Skipping plot.')
                            continue
                        else:
                            # Reset outname for regulatory options
                            outname = "{}.{}.{}.{}.{}.{}.{}".format(grp, plot_type, obsvar+'_reg', startdatename, enddatename, domain_type, domain_name)
                    else:
                        pairdf_reg = None

                    if plot_type.lower() == 'spatial_bias':
                        if use_percentile is None:
                            outname = outname+'.mean'
                        else:
                            outname = outname+'.p'+'{:02d}'.format(use_percentile)

                    if analysis.output_dir is not None:
                        outname = analysis.output_dir + '/' + outname  # Extra / just in case.

                    # Types of plots
                    if plot_type.lower() == 'timeseries' or plot_type.lower() == 'diurnal':
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None
                        # Select time to use as index.

                        # 2024-03-01 MEB needs to only apply if pandas. fails for xarray
                        if isinstance(pairdf,pd.core.frame.DataFrame):
                            pairdf = pairdf.set_index(grp_dict['data_proc']['ts_select_time'])
                        # Specify ts_avg_window if noted in yaml file. #qzr++

                        if 'ts_avg_window' in grp_dict['data_proc'].keys():
                            a_w = grp_dict['data_proc']['ts_avg_window']
                        else:
                            a_w = None

                        #Steps needed to subset paired df if secondary y-axis (altitude_variable) limits are provided,
                        #ELSE: make_timeseries from surfaceplots.py plots the whole df by default
                        #Edit below to accommodate 'ground' or 'mobile' where altitude_yax2 is not needed for timeseries
                        altitude_yax2 = grp_dict['data_proc'].get('altitude_yax2', {})

                        # Extract vmin_y2 and vmax_y2 from filter_dict
                        # Check if 'filter_dict' exists and 'altitude' is a key in filter_criteria
                        # Extract vmin_y2 and vmax_y2 from filter_dict
                        #Better structure for filter_dict (min and max secondary axis) to be optional below
                        filter_criteria = (
                            altitude_yax2.get('filter_dict', None)
                            if isinstance(altitude_yax2, dict)
                            else None
                        )


                        if filter_criteria and 'altitude' in filter_criteria:
                            vmin_y2, vmax_y2 = filter_criteria['altitude']['value']
                        elif filter_criteria is None:

                            if 'altitude' in pairdf:
                                vmin_y2 = pairdf['altitude'].min()
                                vmax_y2 = pairdf['altitude'].max()
                            else:
                                vmin_y2 = vmax_y2 = None
                        else:
                            vmin_y2 = vmax_y2 = None



                        # Check if filter_criteria exists and is not None (Subset the data based on filter criteria if provided)
                        if filter_criteria:
                            for column, condition in filter_criteria.items():
                                operation = condition['oper']
                                value = condition['value']

                                if operation == "between" and isinstance(value, list) and len(value) == 2:
                                    pairdf = pairdf[pairdf[column].between(vmin_y2, vmax_y2)]

                        # Now proceed with plotting, call the make_timeseries function with the subsetted pairdf (if vmin2 and vmax2 are not nOne) otherwise whole df
                        if analysis.obs[p.obs].sat_type is not None and analysis.obs[p.obs].sat_type.startswith("tempo_l2"):
                            if plot_type.lower() == 'timeseries':
                                make_timeseries = xrplots.make_timeseries
                            else:
                                make_timeseries = xrplots.make_diurnal_cycle
                            plot_kwargs = {'dset': pairdf, 'varname': obsvar}
                        else:
                            if plot_type.lower() == "timeseries":
                                make_timeseries = splots.make_timeseries
                            else:
                                make_timeseries = splots.make_diurnal_cycle
                            plot_kwargs = {
                                'df': pairdf, 'df_reg': pairdf_reg, 'column': obsvar
                            }
                        settings = grp_dict.get('settings', {})
                        plot_kwargs = {
                            **plot_kwargs,
                            **{
                                'label':p.obs,
                                'avg_window': a_w,
                                'ylabel': use_ylabel,
                                'vmin':vmin,
                                'vmax':vmax,
                                'domain_type': domain_type,
                                'domain_name': domain_name,
                                'plot_dict': obs_dict,
                                'fig_dict': fig_dict,
                                'text_dict': text_dict,
                                'debug': analysis.debug,
                            },
                            **settings
                        }
                        if p_index == 0:
                            # First plot the observations.
                            ax = make_timeseries(**plot_kwargs)
                        # For all p_index plot the model.
                        if analysis.obs[p.obs].sat_type is not None and analysis.obs[p.obs].sat_type.startswith("tempo_l2"):
                            plot_kwargs['varname']=modvar
                        else:
                            plot_kwargs['column']=modvar
                        plot_kwargs['label'] = p.model
                        plot_kwargs['plot_dict'] = plot_dict
                        plot_kwargs['ax'] = ax
                        ax = make_timeseries(**plot_kwargs)

                        # Extract text_kwargs from the appropriate plot group
                        text_kwargs = grp_dict.get('text_kwargs', {'fontsize': 20})  # Default to fontsize 20 if not defined

                        # At the end save the plot.
                        if p_index == len(pair_labels) - 1:
                            # Adding Altitude variable as secondary y-axis to timeseries (for, model vs aircraft) qzr++
                            if 'altitude_yax2' in grp_dict['data_proc'] and 'altitude_variable' in grp_dict['data_proc']['altitude_yax2']:
                                altitude_yax2 = grp_dict['data_proc']['altitude_yax2']
                                ax = airplots.add_yax2_altitude(ax, pairdf, altitude_yax2, text_kwargs, vmin_y2, vmax_y2)
                            savefig(outname + '.png', logo_height=150)

                            del (ax, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)  # Clear axis for next plot.



                        # At the end save the plot.
                        ##if p_index == len(pair_labels) - 1:
                            #Adding Altitude variable as secondary y-axis to timeseries (for, model vs aircraft) qzr++

                            #Older approach without 'altitude_yax2' control list in YAML now commented out
                            ##if grp_dict['data_proc'].get('altitude_variable'):
                              ##  altitude_variable = grp_dict['data_proc']['altitude_variable']
                              ##  altitude_ticks = grp_dict['data_proc'].get('altitude_ticks', 1000)  # Get altitude tick interval from YAML or default to 1000
                              ##  ax = airplots.add_yax2_altitude(ax, pairdf, altitude_variable, altitude_ticks, text_kwargs)
                            ##savefig(outname + '.png', logo_height=150)
                            ##del (ax, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) #Clear axis for next plot.

                    elif plot_type.lower() == 'curtain':
                        # Set cmin and cmax from obs_plot_dict for colorbar limits
                        if set_yaxis:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                cmin = obs_plot_dict['vmin_plot']
                                cmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                cmin = None
                                cmax = None
                        else:
                            cmin = None
                            cmax = None

                        # Set vmin and vmax from grp_dict for altitude limits
                        if set_yaxis:
                            vmin = grp_dict.get('vmin', None)
                            vmax = grp_dict.get('vmax', None)
                        else:
                            vmin = None
                            vmax = None


                        curtain_config = grp_dict # Curtain plot grp YAML dict
                        # Inside your loop for processing each pair
                        obs_label = p.obs
                        model_label = p.model


                        #Ensure we use the correct observation and model objects from pairing
                        obs = analysis.obs[p.obs]
                        mod = analysis.models[p.model]
                        model_obj = mod.obj

                        # Fetch the observation configuration for colorbar labels
                        obs_label_config = analysis.control_dict['obs'][obs_label]['variables']

                        # Fetch the model and observation data from pairdf
                        pairdf = pairdf_all.reset_index()

                        #### For model_data_2d for curtain/contourfill plot #####
                        # Convert to get something useful for MONET
                        new_ds_obs = obs.obj.rename_axis('time_obs').reset_index().monet._df_to_da().set_coords(['time_obs', 'pressure_obs'])

                        # Nearest neighbor approach to find closest grid cell to each point
                        ds_model = m.util.combinetool.combine_da_to_da(model_obj, new_ds_obs, merge=False)

                        # Interpolate based on time in the observations
                        ds_model = ds_model.interp(time=ds_model.time_obs.squeeze())

                        # Print ds_model and pressure_model values #Debugging
                        ##print(f"ds_model: {ds_model}")
                        ##print(f"pressure_model values: {ds_model['pressure_model'].values}")

                        # Define target pressures for interpolation based on the range of pressure_model
                        min_pressure = ds_model['pressure_model'].min().compute()
                        max_pressure = ds_model['pressure_model'].max().compute()

                        # Fetch the interval and num_levels from curtain_config
                        interval = curtain_config.get('interval', 10000)  # Default to 10,000 Pa if not provided      # Y-axis tick interval
                        num_levels = curtain_config.get('num_levels', 100)   # Default to 100 levels if not provided

                        print(f"Pressure MIN:{min_pressure}, max: {max_pressure}, ytick_interval: {interval}, interpolation_levels: {num_levels}  ")

                        # Use num_levels to define target_pressures interpolation levels
                        target_pressures = np.linspace(max_pressure, min_pressure, num_levels)

                        # Debugging: print target pressures
                        ##print(f"Generated target pressures: {target_pressures}, shape: {target_pressures.shape}")

                        # Check for NaN values before interpolation
                        ##print(f"NaNs in model data before interpolation: {np.isnan(ds_model[modvar]).sum().compute()}")
                        ##print(f"NaNs in pressure_model before interpolation: {np.isnan(ds_model['pressure_model']).sum().compute()}")


                        # Resample model data to target pressures using stratify
                        da_wrf_const = resample_stratify(ds_model[modvar], target_pressures, ds_model['pressure_model'], axis=1, interpolation='linear', extrapolation='nan')
                        da_wrf_const.name = modvar

                        # Create target_pressures DataArray
                        da_target_pressures = xr.DataArray(target_pressures, dims=('z'))
                        da_target_pressures.name = 'target_pressures'

                        # Merge DataArrays into a single Dataset
                        ds_wrf_const = xr.merge([da_wrf_const, da_target_pressures])
                        ds_wrf_const = ds_wrf_const.set_coords('target_pressures')

                        # Debugging: print merged dataset for model curtain
                        ##print(ds_wrf_const)

                        # Ensure model_data_2d is properly reshaped for the contourfill plot
                        model_data_2d = ds_wrf_const[modvar].squeeze()

                        # Debugging: print reshaped model data shape
                        ##print(f"Reshaped model data shape: {model_data_2d.shape}")

                        #### model_data_2d for curtain plot ready ####


                        # Fetch model pressure and other model and observation data from "pairdf" (for scatter plot overlay)
                        time = pairdf['time']
                        obs_pressure = pairdf['pressure_obs']
                        ##print(f"Length of time: {len(time)}") #Debugging
                        ##print(f"Length of obs_pressure: {len(obs_pressure)}") #Debugging

                        # Generate the curtain plot using airplots.make_curtain_plot
                        try:
                            outname_pair = f"{outname}_{obs_label}_vs_{model_label}.png"

                            print(f"Saving curtain plot to {outname_pair}...")

                            ax = airplots.make_curtain_plot(
                                time=pd.to_datetime(time),
                                altitude=target_pressures,  # Use target_pressures for interpolation
                                model_data_2d=model_data_2d,  # Already reshaped to match the expected shape
                                obs_pressure=obs_pressure,  # Pressure_obs for obs scatter plot
                                pairdf=pairdf,  #use pairdf for scatter overlay (model and obs)
                                mod_var=modvar,
                                obs_var=obsvar,
                                grp_dict=curtain_config,
                                vmin=vmin,
                                vmax=vmax,
                                cmin=cmin,
                                cmax=cmax,
                                plot_dict=plot_dict,
                                outname=outname_pair,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                obs_label_config=obs_label_config,
                                text_dict=text_dict,
                                debug=analysis.debug  # Pass debug flag
                            )


                        except Exception as e:
                            print(f"Error generating curtain plot for {modvar} vs {obsvar}: {e}")
                        finally:
                            plt.close('all')  # Clean up matplotlib resources




                    #qzr++ Added vertprofile plotype for aircraft vs model comparisons
                    elif plot_type.lower() == 'vertprofile':
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None
                        # Select altitude variable from the .yaml file
                        altitude_variable = grp_dict['altitude_variable']
                        # Define the bins for binning the altitude
                        bins = grp_dict['vertprofile_bins']
                        if p_index == 0:
                            # First plot the observations.
                            ax = airplots.make_vertprofile(
                                pairdf,
                                column=obsvar,
                                label=p.obs,
                                bins=bins,
                                altitude_variable=altitude_variable,
                                ylabel=use_ylabel,
                                vmin=vmin,
                                vmax=vmax,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                plot_dict=obs_dict,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug,
                                interquartile_style=interquartile_style
                        )

                        # For all p_index plot the model.
                        ax = airplots.make_vertprofile(
                            pairdf,
                            column=modvar,
                            label=p.model,
                            ax=ax,
                            bins=bins,
                            altitude_variable=altitude_variable,
                            ylabel=use_ylabel,
                            vmin=vmin,
                            vmax=vmax,
                            domain_type=domain_type,
                            domain_name=domain_name,
                            plot_dict=plot_dict,
                            text_dict=text_dict,
                            debug=analysis.debug,
                            interquartile_style=interquartile_style
                        )


                        # At the end save the plot.
                        if p_index == len(pair_labels) - 1:
                            savefig(outname + '.png', logo_height=250)
                            del (ax, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) # Clear axis for next plot.

                    elif plot_type.lower() == 'vertical_single_date':
                        #to use vmin, vmax from obs in yaml
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot','vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('warning: vmin_plot and vmax_plot not specified for '+obsvar+',so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None
                        #begin plotting
                        if p_index ==0:
                            comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=obsvar, label=p.obs, plot_dict=obs_dict)
                        comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=modvar, label=p.model, plot_dict=plot_dict, comb_bx = comb_bx, label_bx = label_bx)
                        if p_index == len(pair_labels) - 1:
                            sondeplots.make_vertical_single_date(pairdf,
                                                                  comb_bx,
                                                                  altitude_range=altitude_range,
                                                                  altitude_method=altitude_method,
                                                                  vmin=vmin,
                                                                  vmax=vmax,
                                                                  station_name=station_name,
                                                                  release_time=release_time,
                                                                  label_bx=label_bx,
                                                                  fig_dict=fig_dict,
                                                                  text_dict=text_dict
                                                                  )
                            #save plot
                            plt.tight_layout()
                            savefig(outname+".png", loc=monet_logo_position[0], logo_height=100, dpi=300)

                            del (comb_bx,label_bx,fig_dict, plot_dict, text_dict, obs_dict,obs_plot_dict)

                    elif plot_type.lower() == 'vertical_boxplot_os':
                        #to use vmin, vmax from obs in yaml
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot','vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('warning: vmin_plot and vmax_plot not specified for '+obsvar+',so default used.')
                                vmin=None
                                vmax=None
                        else:
                            vmin=None
                            vmax=None
                        #begin plotting
                        if p_index ==0:
                            comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=obsvar, label=p.obs, plot_dict=obs_dict)
                        comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=modvar, label=p.model, plot_dict=plot_dict, comb_bx = comb_bx, label_bx = label_bx)

                        if p_index == len(pair_labels) - 1:
                            sondeplots.make_vertical_boxplot_os(pairdf,
                                                                 comb_bx,
                                                                 label_bx=label_bx,
                                                                 altitude_range=altitude_range,
                                                                 altitude_method=altitude_method,
                                                                 vmin=vmin,
                                                                 vmax=vmax,
                                                                 altitude_threshold_list=altitude_threshold_list,
                                                                 station_name=station_name,
                                                                 release_time=release_time,
                                                                 fig_dict=fig_dict,
                                                                 text_dict=text_dict)
                            #save plot
                            plt.tight_layout()
                            savefig(outname+".png", loc=monet_logo_position[0], logo_height=100, dpi=300)
                            del (comb_bx,label_bx,fig_dict, plot_dict, text_dict, obs_dict,obs_plot_dict)

                    elif plot_type.lower() == 'density_scatter_plot_os':
                        #to use vmin, vmax from obs in yaml
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot','vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('warning: vmin_plot and vmax_plot not specified for '+obsvar+',so default used.')
                                vmin=None
                                vmax=None
                        else:
                            vmin=None
                            vmax=None

                        #begin plotting
                        plt.figure()
                        sondeplots.density_scatter_plot_os(pairdf,altitude_range,vmin,vmax,station_name,altitude_method,cmap_method,modvar,obsvar)
                        plt.title('Scatter plot for '+model_name_list[0]+' vs. '+model_name_list[p_index+1]+'\nat '+str(station_name[0])+' on '+str(release_time)+' UTC',fontsize=15)
                        plt.tight_layout()
                        savefig(outname+"."+p_label+"."+"-".join(altitude_method[0].split())+".png", loc=monet_logo_position[0], logo_height=100, dpi=300)
                        del (pairdf)

                    elif plot_type.lower() == 'violin':
                        if set_yaxis:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None

                        # Initialize the combined DataFrame for violin plots and labels/colors list
                        if p_index == 0:
                            comb_violin = pd.DataFrame()
                            label_violin = []


                        # Define a default color for observations
                        default_obs_color = 'gray'  # Default color for observations

                        # Inside your loop for processing each pair
                        obs_label = p.obs
                        model_label = p.model

                        # Retrieve plot_kwargs for observation
                        if hasattr(analysis.obs[p.obs], 'plot_kwargs') and analysis.obs[p.obs].plot_kwargs is not None:
                            obs_dict = analysis.obs[p.obs].plot_kwargs
                        else:
                            obs_dict = {'color': default_obs_color}

                        # Retrieve plot_kwargs for the model
                        model_dict = analysis.models[p.model].plot_kwargs if analysis.models[p.model].plot_kwargs is not None else {'color': 'blue'} # Fallback color for models, in case it's missing

                        # Call calculate_violin for observation data
                        if p_index ==0:
                            comb_violin, label_violin = airplots.calculate_violin(
                                df=pairdf,
                                column=obsvar,
                                label=obs_label,
                                plot_dict=obs_dict,
                                comb_violin=comb_violin,
                                label_violin=label_violin
                            )

                        # Call calculate_violin for model data
                        comb_violin, label_violin = airplots.calculate_violin(
                            df=pairdf,
                            column=modvar,
                            label=model_label,
                            plot_dict=model_dict,
                            comb_violin=comb_violin,
                            label_violin=label_violin
                        )


                        # For the last pair, create the violin plot
                        if p_index == len(pair_labels) - 1:
                            airplots.make_violin_plot(
                                comb_violin=comb_violin,
                                label_violin=label_violin,
                                ylabel=use_ylabel,
                                vmin=vmin,
                                vmax=vmax,
                                outname=outname,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug
                            )

                        # Clear the variables for the next plot if needed
                        if p_index == len(pair_labels) - 1:
                            del (comb_violin, label_violin, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)



                    elif plot_type.lower() == 'scatter_density':
                        scatter_density_config = grp_dict


                        # Extract relevant parameters from the configuration
                        color_map = scatter_density_config.get('color_map', 'viridis')
                        fill = scatter_density_config.get('fill', False)
                        print(f"Value of fill after reading from scatter_density_config: {fill}") #Debugging


                        vmin_x = scatter_density_config.get('vmin_x', None)
                        vmax_x = scatter_density_config.get('vmax_x', None)
                        vmin_y = scatter_density_config.get('vmin_y', None)
                        vmax_y = scatter_density_config.get('vmax_y', None)

                        # Accessing the correct model and observation configuration/labels/variables
                        model_label = p.model
                        obs_label = p.obs

                        try:
                            _ = analysis.control_dict['model'][model_label]['mapping'][obs_label]
                        except KeyError:
                            print(f"Error: Mapping not found for model label '{model_label}' with observation label '{obs_label}' in scatter_density plot")
                            continue  # Skip this iteration if mapping is not found

                        obs_config = analysis.control_dict['obs'][obs_label]['variables'] # Accessing the correct observation configuration


                        # Extract ylabel_plot for units extraction
                        ylabel_plot = obs_config.get(obsvar, {}).get('ylabel_plot', f"{obsvar} (units)")
                        title = ylabel_plot
                        units = ylabel_plot[ylabel_plot.find("(")+1 : ylabel_plot.find(")")]
                        xlabel = f"Model {modvar} ({units})"
                        ylabel = f"Observation {obsvar} ({units})"




                        # Exclude keys from kwargs that are being passed explicitly
                        excluded_keys = ['color_map', 'fill', 'vmin_x', 'vmax_x', 'vmin_y', 'vmax_y', 'xlabel', 'ylabel', 'title', 'data']
                        kwargs = {key: value for key, value in scatter_density_config.items() if key not in excluded_keys}
                        if 'shade_lowest' in kwargs:
                            kwargs['thresh'] = 0
                            del kwargs['shade_lowest']


                        outname_pair = f"{outname}_{obs_label}_vs_{model_label}.png"

                        print(f"Saving scatter density plot to {outname_pair}...")

                        # Create the scatter density plot
                        print(f"Processing scatter density plot for model '{model_label}' and observation '{obs_label}'...")
                        ax = airplots.make_scatter_density_plot(
                            pairdf,
                            mod_var=modvar,
                            obs_var=obsvar,
                            color_map=color_map,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            title=title,
                            fill=fill,
                            vmin_x=vmin_x,
                            vmax_x=vmax_x,
                            vmin_y=vmin_y,
                            vmax_y=vmax_y,
                            outname=outname_pair,
                            **kwargs
                        )


                        plt.close()  # Close the current figure

                    elif plot_type.lower() == 'boxplot':
                        # squeeze the xarray for boxplot, M.Li
                        if obs_type in ["sat_swath_sfc", "sat_swath_clm",
                                        "sat_grid_sfc", "sat_grid_clm",
                                        "sat_swath_prof"]:
                            pairdf_sel = pairdf.squeeze()
                        else:
                            pairdf_sel = pairdf

                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None
                        # First for p_index = 0 create the obs box plot data array.
                        if p_index == 0:
                            comb_bx, label_bx = splots.calculate_boxplot(pairdf_sel, pairdf_reg, column=obsvar,
                                                                                   label=p.obs, plot_dict=obs_dict)
                        # Then add the models to this dataarray.
                        comb_bx, label_bx = splots.calculate_boxplot(pairdf_sel, pairdf_reg, column=modvar, label=p.model,
                                                                                plot_dict=plot_dict, comb_bx=comb_bx,
                                                                                label_bx=label_bx)
                        # For the last p_index make the plot.
                        if p_index == len(pair_labels) - 1:
                            splots.make_boxplot(
                                comb_bx,
                                label_bx,
                                ylabel=use_ylabel,
                                vmin=vmin,
                                vmax=vmax,
                                outname=outname,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                plot_dict=obs_dict,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug
                            )
                            #Clear info for next plot.
                            del (comb_bx, label_bx, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)

                    elif plot_type.lower() == 'multi_boxplot':
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                        else:
                            vmin = None
                            vmax = None
                        # First for p_index = 0 create the obs box plot data array.

                        if p_index == 0:
                            comb_bx, label_bx,region_bx = splots.calculate_multi_boxplot(pairdf, pairdf_reg,region_name=region_name, column=obsvar,
                                                                         label=p.obs, plot_dict=obs_dict)

                        # Then add the models to this dataarray.
                        comb_bx, label_bx,region_bx = splots.calculate_multi_boxplot(pairdf, pairdf_reg, region_name= region_name,column=modvar, label=p.model,
                                                                     plot_dict=plot_dict, comb_bx=comb_bx,
                                                                     label_bx=label_bx)

                        # For the last p_index make the plot.
                        if p_index == len(pair_labels) - 1:
                            splots.make_multi_boxplot(
                                comb_bx,
                                label_bx,
                                region_bx,
                                region_list = region_list,
                                model_name_list=model_name_list,
                                ylabel=use_ylabel,
                                vmin=vmin,
                                vmax=vmax,
                                outname=outname,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                plot_dict=obs_dict,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug)
                            #Clear info for next plot.
                            del (comb_bx, label_bx,region_bx, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)

                    elif plot_type.lower() == 'scorecard':
                        # First for p_index = 0 create the obs box plot data array.
                        if p_index == 0:
                            comb_bx, label_bx,region_bx,msa_bx,time_bx = splots.scorecard_step1_combine_df(pairdf, pairdf_reg,region_name=region_name,urban_rural_name=urban_rural_name,
                                                                                                   column=obsvar, label=p.obs, plot_dict=obs_dict)
                        # Then add the model to this dataarray.
                        comb_bx, label_bx,region_bx, msa_bx,time_bx = splots.scorecard_step1_combine_df(pairdf, pairdf_reg, region_name= region_name,urban_rural_name=urban_rural_name,
                                                                                               column=modvar, label=p.model, plot_dict=plot_dict, comb_bx=comb_bx, label_bx=label_bx)
                        # For the last p_index make the plot.
                        if p_index == len(pair_labels) - 1:
                            output_obs, output_model1, output_model2 = splots.scorecard_step2_prepare_individual_df(comb_bx,region_bx,msa_bx,time_bx,model_name_list=model_name_list)

                            #split by region, data, and urban/rural
                            datelist = splots.GetDateList(analysis.start_time,analysis.end_time)
                            OBS_Region_Date_Urban_list, OBS_Region_Date_Rural_list = splots.scorecard_step4_GetRegionLUCDate(ds_name=output_obs,region_list=region_list,datelist=datelist,urban_rural_differentiate_value=urban_rural_differentiate_value)
                            MODEL1_Region_Date_Urban_list, MODEL1_Region_Date_Rural_list= splots.scorecard_step4_GetRegionLUCDate(ds_name=output_model1,region_list=region_list,datelist=datelist,urban_rural_differentiate_value=urban_rural_differentiate_value)
                            MODEL2_Region_Date_Urban_list, MODEL2_Region_Date_Rural_list= splots.scorecard_step4_GetRegionLUCDate(ds_name=output_model2,region_list=region_list,datelist=datelist,urban_rural_differentiate_value=urban_rural_differentiate_value)

                            #Kick Nan values
                            OBS_Region_Date_Urban_list_noNan,MODEL1_Region_Date_Urban_list_noNan,MODEL2_Region_Date_Urban_list_noNan = splots.scorecard_step5_KickNan(obs_input=OBS_Region_Date_Urban_list,
                                                                                                                                                                      model_input_1=MODEL1_Region_Date_Urban_list,
                                                                                                                                                                      model_input_2=MODEL2_Region_Date_Urban_list)
                            OBS_Region_Date_Rural_list_noNan,MODEL1_Region_Date_Rural_list_noNan,MODEL2_Region_Date_Rural_list_noNan = splots.scorecard_step5_KickNan(obs_input=OBS_Region_Date_Rural_list,
                                                                                                                                                                      model_input_1=MODEL1_Region_Date_Rural_list,
                                                                                                                                                                      model_input_2=MODEL2_Region_Date_Rural_list)
                            #Get final output Matrix
                            Output_matrix = splots.scorecard_step8_OutputMatrix(obs_urban_input    = OBS_Region_Date_Urban_list_noNan,
                                                                                model1_urban_input = MODEL1_Region_Date_Urban_list_noNan,
                                                                                model2_urban_input = MODEL2_Region_Date_Urban_list_noNan,
                                                                                obs_rural_input    = OBS_Region_Date_Rural_list_noNan,
                                                                                model1_rural_input = MODEL1_Region_Date_Rural_list_noNan,
                                                                                model2_rural_input = MODEL2_Region_Date_Rural_list_noNan,
                                                                                better_or_worse_method = better_or_worse_method)
                            #plot the scorecard
                            splots.scorecard_step9_makeplot(output_matrix=Output_matrix,
                                                     column=obsvar,
                                                     region_list=region_list,
                                                     model_name_list=model_name_list,
                                                     outname=outname,
                                                     domain_type=domain_type,
                                                     domain_name=domain_name,
                                                     fig_dict=fig_dict,
                                                     text_dict=text_dict,
                                                     datelist=datelist,
                                                     better_or_worse_method = better_or_worse_method)
                            #Clear info for next plot.
                            del (comb_bx, label_bx, region_bx, msa_bx, time_bx, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)

                    elif plot_type.lower() == 'csi':
                        # First for p_index = 0 create the obs box plot data array.
                        if p_index == 0:

                            comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=obsvar,label=p.obs, plot_dict=obs_dict)
                            print(p_index,np.shape(comb_bx))
                        # Then add the models to this dataarray.
                        comb_bx, label_bx = splots.calculate_boxplot(pairdf, pairdf_reg, column=modvar, label=p.model,plot_dict=plot_dict, comb_bx=comb_bx, label_bx=label_bx)
                        print(p_index,np.shape(comb_bx))
                        if p_index == len(pair_labels) - 1:
                            print('final',p_index, len(pair_labels) - 1)
                            splots.Plot_CSI(column=obsvar,
                                            score_name_input=score_name,
                                            threshold_list_input=threshold_list,
                                            comb_bx_input=comb_bx,
                                            plot_dict=plot_dict,
                                            fig_dict=fig_dict,
                                            text_dict=text_dict,
                                            domain_type=domain_type,
                                            domain_name=domain_name,
                                            model_name_list=model_name_list,
                                            threshold_tick_style=threshold_tick_style)
                            #save figure
                            plt.tight_layout()
                            savefig(outname +'.'+score_name+'.png', loc=1, logo_height=100)

                            #Clear info for next plot.
                            del (comb_bx, label_bx, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict)


                    elif plot_type.lower() == 'taylor':
                        if analysis.obs[p.obs].sat_type is not None and analysis.obs[p.obs].sat_type.startswith("tempo_l2"):
                            make_taylor = xrplots.make_taylor
                            plot_kwargs = {
                                'dset': pairdf,
                                'varname_o': obsvar,
                                'varname_m': modvar,
                                'normalize': True,
                            }
                        else:
                            make_taylor = splots.make_taylor
                            plot_kwargs = {
                                'df': pairdf,
                                'column_o': obsvar,
                                'column_m': modvar,
                            }
                        plot_kwargs = {
                            **plot_kwargs,
                            **{
                                'label_o': p.obs,
                                'label_m': p.model,
                                'ylabel': use_ylabel,
                                'domain_type': domain_type,
                                'domain_name': domain_name,
                                'plot_dict': plot_dict,
                                'fig_dict': fig_dict,
                                'text_dict': text_dict,
                                'debug': analysis.debug,
                            }
                        }

                        if set_yaxis is True:
                            if 'ty_scale' in obs_plot_dict.keys():
                                plot_kwargs["ty_scale"] = obs_plot_dict['ty_scale']
                            else:
                                print('Warning: ty_scale not specified for ' + obsvar + ', so default used.')
                                plot_kwargs["ty_scale"] = 1.5  # Use default
                        else:
                            plot_kwargs["ty_scale"] = 1.5  # Use default
                        try:
                            plot_kwargs["ty_scale"] = grp_dict["data_proc"].get("ty_scale", 1.5)
                        except KeyError:
                            plot_kwargs["ty_scale"]=2
                        if p_index == 0:
                            # Plot initial obs/model
                            dia = make_taylor(**plot_kwargs)
                        else:
                            # For the rest, plot on top of dia
                            dia = make_taylor(dia=dia, **plot_kwargs)
                        # At the end save the plot.
                        if p_index == len(pair_labels) - 1:
                            savefig(outname + '.png', logo_height=70)
                            del (dia, fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) #Clear info for next plot.




                    elif plot_type.lower() == 'spatial_bias':
                        if set_yaxis is True:
                            if 'vdiff_plot' in obs_plot_dict.keys():
                                vdiff = obs_plot_dict['vdiff_plot']
                            else:
                                print('Warning: vdiff_plot not specified for ' + obsvar + ', so default used.')
                                vdiff = None
                        else:
                            vdiff = None
                        # p_label needs to be added to the outname for this plot
                        outname = "{}.{}".format(outname, p_label)
                        splots.make_spatial_bias(
                            pairdf,
                            pairdf_reg,
                            column_o=obsvar,
                            label_o=p.obs,
                            column_m=modvar,
                            label_m=p.model,
                            ylabel=use_ylabel,
                            ptile=use_percentile,
                            vdiff=vdiff,
                            outname=outname,
                            domain_type=domain_type,
                            domain_name=domain_name,
                            fig_dict=fig_dict,
                            text_dict=text_dict,
                            debug=analysis.debug
                        )
                    elif plot_type.lower() == 'gridded_spatial_bias':
                        outname = "{}.{}".format(outname, p_label)
                        if analysis.obs[p.obs].sat_type is not None and analysis.obs[p.obs].sat_type.startswith("tempo_l2"):
                            make_spatial_bias_gridded = xrplots.make_spatial_bias_gridded
                            plot_kwargs = {
                                'dset': pairdf, 'varname_o': obsvar, 'varname_m': modvar,
                            }
                        else:
                            make_spatial_bias_gridded = splots.make_spatial_bias_gridded
                            plot_kwargs = {'df': pairdf, 'column_o': obsvar, 'column_m': modvar}

                        plot_kwargs = {
                            **plot_kwargs,
                            **{
                                "label_o": p.obs,
                                "label_m": p.model,
                                "ylabel": use_ylabel,
                                "outname": outname,
                                "domain_type": domain_type,
                                "domain_name": domain_name,
                                "vdiff": grp_dict["data_proc"].get("vdiff", None),
                                "vmax": grp_dict["data_proc"].get("vmax", None),
                                "vmin": grp_dict["data_proc"].get("vmin", None),
                                "nlevels": grp_dict["data_proc"].get("nlevels", None),
                                "fig_dict": fig_dict,
                                "text_dict": text_dict,
                                "debug": analysis.debug
                            }
                        }
                        make_spatial_bias_gridded(**plot_kwargs)
                        del (fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) #Clear info for next plot.
                    elif plot_type.lower() == 'spatial_dist':
                        outname = "{}.{}".format(outname, p.obs)
                        plot_kwargs = {
                            "dset": pairdf,
                            "varname": obsvar,
                            "outname": outname,
                            "label": p.obs,
                            "ylabel": use_ylabel,
                            "domain_type": domain_type,
                            "domain_name": domain_name,
                            "vmax": grp_dict["data_proc"].get("vmax", None),
                            "vmin": grp_dict["data_proc"].get("vmin", None),
                            "fig_dict": fig_dict,
                            "text_dict": text_dict,
                            "debug": analysis.debug,
                        }
                        if isinstance(plot_kwargs["vmax"], str):
                            plot_kwargs["vmax"] = float(plot_kwargs["vmax"])
                        if isinstance(plot_kwargs["vmin"], str):
                            plot_kwargs["vmin"] = float(plot_kwargs["vmin"])
                        xrplots.make_spatial_dist(**plot_kwargs)
                        plot_kwargs["varname"] = modvar
                        plot_kwargs["label"] = p.model
                        plot_kwargs["outname"] = outname.replace(p.obs, p.model)
                        xrplots.make_spatial_dist(**plot_kwargs)
                    elif plot_type.lower() == 'spatial_bias_exceedance':
                        if cal_reg:
                            if set_yaxis is True:
                                if 'vdiff_reg_plot' in obs_plot_dict.keys():
                                    vdiff = obs_plot_dict['vdiff_reg_plot']
                                else:
                                    print('Warning: vdiff_reg_plot not specified for ' + obsvar + ', so default used.')
                                    vdiff = None
                            else:
                                vdiff = None

                            # p_label needs to be added to the outname for this plot
                            outname = "{}.{}".format(outname, p_label)
                            splots.make_spatial_bias_exceedance(
                                pairdf_reg,
                                column_o=obsvar+'_reg',
                                label_o=p.obs,
                                column_m=modvar+'_reg',
                                label_m=p.model,
                                ylabel=use_ylabel,
                                vdiff=vdiff,
                                outname=outname,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug
                            )
                            del (fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) #Clear info for next plot.
                        else:
                            print('Warning: spatial_bias_exceedance plot only works when regulatory=True.')
                    # JianHe: need updates to include regulatory option for overlay plots
                    elif plot_type.lower() == 'spatial_overlay':
                        if set_yaxis is True:
                            if all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot', 'nlevels_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                                nlevels = obs_plot_dict['nlevels_plot']
                            elif all(k in obs_plot_dict for k in ('vmin_plot', 'vmax_plot')):
                                vmin = obs_plot_dict['vmin_plot']
                                vmax = obs_plot_dict['vmax_plot']
                                nlevels = None
                            else:
                                print('Warning: vmin_plot and vmax_plot not specified for ' + obsvar + ', so default used.')
                                vmin = None
                                vmax = None
                                nlevels = None
                        else:
                            vmin = None
                            vmax = None
                            nlevels = None
                        #Check if z dim is larger than 1. If so select, the first level as all models read through
                        #MONETIO will be reordered such that the first level is the level nearest to the surface.
                        # Create model slice and select time window for spatial plots
                        try:
                            analysis.models[p.model].obj.sizes['z']
                            if analysis.models[p.model].obj.sizes['z'] > 1: #Select only surface values.
                                vmodel = analysis.models[p.model].obj.isel(z=0).expand_dims('z',axis=1).loc[
                                    dict(time=slice(analysis.start_time, analysis.end_time))]
                            else:
                                vmodel = analysis.models[p.model].obj.loc[dict(time=slice(analysis.start_time, analysis.end_time))]
                        except KeyError as e:
                            raise Exception("MONET requires an altitude dimension named 'z'") from e
                        if grp_dict.get('data_proc', {}).get('crop_model', False) and domain_name != all:
                            vmodel = select_region(vmodel, domain_type, domain_name, domain_info)

                        # Determine proj to use for spatial plots
                        proj = splots.map_projection(analysis.models[p.model])
                        # p_label needs to be added to the outname for this plot
                        outname = "{}.{}".format(outname, p_label)
                        # For just the spatial overlay plot, you do not use the model data from the pair file
                        # So get the variable name again since pairing one could be _new.
                        # JianHe: only make overplay plots for non-regulatory variables for now
                        if not cal_reg:
                            splots.make_spatial_overlay(
                                pairdf,
                                vmodel,
                                column_o=obsvar,
                                label_o=p.obs,
                                column_m=p.model_vars[index],
                                label_m=p.model,
                                ylabel=use_ylabel,
                                vmin=vmin,
                                vmax=vmax,
                                nlevels=nlevels,
                                proj=proj,
                                outname=outname,
                                domain_type=domain_type,
                                domain_name=domain_name,
                                fig_dict=fig_dict,
                                text_dict=text_dict,
                                debug=analysis.debug
                            )
                        else:
                            print('Warning: Spatial overlay plots are not available yet for regulatory metrics.')

                        del (fig_dict, plot_dict, text_dict, obs_dict, obs_plot_dict) #Clear info for next plot.

    # Restore figure count warning
    plt.rcParams["figure.max_open_warning"] = initial_max_fig
