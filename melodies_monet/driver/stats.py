# SPDX-License-Identifier: Apache-2.0
#
"""
The stats module for MELODIES-MONET.
"""
import datetime
import pandas as pd

def stats(analysis):
    """Calculate statistics specified in the input yaml file.

    This routine  loops over all the domains and model/obs pairs for all the variables
    specified in the mapping dictionary listed in :attr:`paired`.

    Creates a csv file storing the statistics and optionally a figure
    visualizing the table.

    Parameters
    ----------
    analysis : melodies_monet.driver.analysis
        The analysis object.

    Returns
    -------
    None
    """
    from ..stats import proc_stats as proc_stats
    from ..plots import surfplots as splots
    from ..util.region_select import select_region

    # first get the stats dictionary from the yaml file
    stat_dict = analysis.control_dict['stats']
    # Calculate general items
    startdatename = str(datetime.datetime.strftime(analysis.start_time, '%Y-%m-%d_%H'))
    enddatename = str(datetime.datetime.strftime(analysis.end_time, '%Y-%m-%d_%H'))
    stat_list = stat_dict['stat_list']
    # Determine stat_grp full name
    stat_fullname_ns = proc_stats.produce_stat_dict(stat_list=stat_list, spaces=False)
    stat_fullname_s = proc_stats.produce_stat_dict(stat_list=stat_list, spaces=True)
    pair_labels = stat_dict['data']

    # Determine rounding
    if 'round_output' in stat_dict.keys():
        round_output = stat_dict['round_output']
    else:
        round_output = 3

    # Then loop over all the observations
    # first get the observational obs labels
    pair1 = analysis.paired[list(analysis.paired.keys())[0]]
    obs_vars = pair1.obs_vars
    for obsvar in obs_vars:
        # Read in some plotting specifications stored with observations.
        if analysis.obs[pair1.obs].variable_dict is not None:
            if obsvar in analysis.obs[pair1.obs].variable_dict.keys():
                obs_plot_dict = analysis.obs[pair1.obs].variable_dict[obsvar]
            else:
                obs_plot_dict = {}
        else:
            obs_plot_dict = {}

        # JianHe: Determine if calculate regulatory values
        cal_reg = obs_plot_dict.get('regulatory', False)

        # Next loop over all of the domains.
        # Loop also over the domain types.
        domain_types = stat_dict['domain_type']
        domain_names = stat_dict['domain_name']
        domain_infos = stat_dict.get('domain_info', {})
        for domain in range(len(domain_types)):
            domain_type = domain_types[domain]
            domain_name = domain_names[domain]
            domain_info = domain_infos.get(domain_name, None)

            # The tables and text files will be output at this step in loop.
            # Create an empty pandas dataarray.
            df_o_d = pd.DataFrame()
            # Determine outname
            if cal_reg:
                outname = "{}.{}.{}.{}.{}.{}".format('stats', obsvar+'_reg', domain_type, domain_name, startdatename, enddatename)
            else:
                outname = "{}.{}.{}.{}.{}.{}".format('stats', obsvar, domain_type, domain_name, startdatename, enddatename)

            # Determine plotting kwargs
            if 'output_table_kwargs' in stat_dict.keys():
                out_table_kwargs = stat_dict['output_table_kwargs']
            else:
                out_table_kwargs = None

            # Add Stat ID and FullName to pandas dictionary.
            df_o_d['Stat_ID'] = stat_list
            df_o_d['Stat_FullName'] = stat_fullname_ns

            # Specify title for stat plots.
            if cal_reg:
                if 'ylabel_reg_plot' in obs_plot_dict.keys():
                    title = obs_plot_dict['ylabel_reg_plot'] + ': ' + domain_type + ' ' + domain_name
                else:
                    title = obsvar + '_reg: ' + domain_type + ' ' + domain_name
            else:
                if 'ylabel_plot' in obs_plot_dict.keys():
                    title = obs_plot_dict['ylabel_plot'] + ': ' + domain_type + ' ' + domain_name
                else:
                    title = obsvar + ': ' + domain_type + ' ' + domain_name

            # Finally Loop through each of the pairs
            for p_label in pair_labels:
                p = analysis.paired[p_label]
                # Create an empty list to store the stat_var
                p_stat_list = []

                # Loop through each of the stats
                for stat_grp in stat_list:

                    # find the pair model label that matches the obs var
                    index = p.obs_vars.index(obsvar)
                    modvar = p.model_vars[index]

                    # Adjust the modvar as done in pairing script, if the species name in obs and model are the same.
                    if obsvar == modvar:
                        modvar = modvar + '_new'
                    # for satellite no2 trop. columns paired data, M.Li
                    if obsvar == 'nitrogendioxide_tropospheric_column':
                        modvar = modvar + 'trpcol'

                    # Query selected points if applicable
                    if domain_type != 'all':
                        p_region = select_region(p.obj, domain_type, domain_name, domain_info)
                    else:
                        p_region = p.obj

                    dim_order = [dim for dim in ["time", "y", "x"] if dim in p_region.dims]
                    pairdf_all = p_region.to_dataframe(dim_order=dim_order)

                    # Select only the analysis time window.
                    pairdf_all = pairdf_all.loc[analysis.start_time : analysis.end_time]

                    # Query with filter options
                    if 'data_proc' in stat_dict:
                        if 'filter_dict' in stat_dict['data_proc'] and 'filter_string' in stat_dict['data_proc']:
                            raise Exception("For statistics, only one of filter_dict and filter_string can be specified.")
                        elif 'filter_dict' in stat_dict['data_proc']:
                            filter_dict = stat_dict['data_proc']['filter_dict']
                            for column in filter_dict.keys():
                                filter_vals = filter_dict[column]['value']
                                filter_op = filter_dict[column]['oper']
                                if filter_op == 'isin':
                                    pairdf_all.query(f'{column} == {filter_vals}', inplace=True)
                                elif filter_op == 'isnotin':
                                    pairdf_all.query(f'{column} != {filter_vals}', inplace=True)
                                else:
                                    pairdf_all.query(f'{column} {filter_op} {filter_vals}', inplace=True)
                        elif 'filter_string' in stat_dict['data_proc']:
                            pairdf_all.query(stat_dict['data_proc']['filter_string'], inplace=True)

                    # Drop sites with greater than X percent NAN values
                    if 'data_proc' in stat_dict:
                        if 'rem_obs_by_nan_pct' in stat_dict['data_proc']:
                            grp_var = stat_dict['data_proc']['rem_obs_by_nan_pct']['group_var']
                            pct_cutoff = stat_dict['data_proc']['rem_obs_by_nan_pct']['pct_cutoff']

                            if stat_dict['data_proc']['rem_obs_by_nan_pct']['times'] == 'hourly':
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

                    # Drop NaNs for model and observations in all cases.
                    pairdf = pairdf_all.reset_index().dropna(subset=[modvar, obsvar])

                    # JianHe: do we need provide a warning if pairdf is empty (no valid obsdata) for specific subdomain?
                    if pairdf[obsvar].isnull().all() or pairdf.empty:
                        print('Warning: no valid obs found for '+domain_name)
                        p_stat_list.append('NaN')
                        continue

                    if cal_reg:
                        # Process regulatory values
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
                            print('Warning: no regulatory calculations found for ' + obsvar + '. Setting stat calculation to NaN.')
                            del df2
                            p_stat_list.append('NaN')
                            continue
                        del df2
                        if len(pairdf_reg[obsvar+'_reg']) == 0:
                            print('No valid data for '+obsvar+'_reg. Setting stat calculation to NaN.')
                            p_stat_list.append('NaN')
                            continue
                        else:
                            # Drop NaNs for model and observations in all cases.
                            pairdf2 = pairdf_reg.reset_index().dropna(subset=[modvar+'_reg', obsvar+'_reg'])

                    # Create empty list for all dom
                    # Calculate statistic and append to list
                    if obsvar == 'WD':  # Use separate calculations for WD
                        p_stat_list.append(proc_stats.calc(pairdf, stat=stat_grp, obsvar=obsvar, modvar=modvar, wind=True))
                    else:
                        if cal_reg:
                            p_stat_list.append(proc_stats.calc(pairdf2, stat=stat_grp, obsvar=obsvar+'_reg', modvar=modvar+'_reg', wind=False))
                        else:
                            p_stat_list.append(proc_stats.calc(pairdf, stat=stat_grp, obsvar=obsvar, modvar=modvar, wind=False))

                # Save the stat to a dataarray
                df_o_d[p_label] = p_stat_list

            if analysis.output_dir is not None:
                outname = analysis.output_dir + '/' + outname  # Extra / just in case.

            # Save the pandas dataframe to a txt file
            # Save rounded output
            df_o_d = df_o_d.round(round_output)
            df_o_d.to_csv(path_or_buf=outname + '.csv', index=False)

            if stat_dict['output_table'] is True:
                # Output as a table graphic too.
                # Change to use the name with full spaces.
                df_o_d['Stat_FullName'] = stat_fullname_s

                proc_stats.create_table(df_o_d.drop(columns=['Stat_ID']),
                                        outname=outname,
                                        title=title,
                                        out_table_kwargs=out_table_kwargs,
                                        debug=analysis.debug
                                       )
