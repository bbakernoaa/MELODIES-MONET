# SPDX-License-Identifier: Apache-2.0
#
"""
Pairing strategies for different observation types.
"""
import pandas as pd

from ..util import satellite_utilities as sutil
from ..util import sat_l2_swath_utility as no2util
from ..util import sat_l2_swath_utility_tempo as tempo_sutil
from ..util.tools import vert_interp, mobile_and_ground_pair
from .pair import pair


def _pair_pt_sfc(model_obj, mod, obs, keys, obs_vars, debug):
    """Pair point surface observations.
    """
    if not isinstance(obs.obj, pd.DataFrame):
        obs.obs_to_df()

    model_obj_for_pairing = model_obj
    try:
        if model_obj.sizes['z'] > 1:
            model_obj_for_pairing = model_obj.isel(z=0).expand_dims('z', axis=1)
    except KeyError as e:
        raise Exception("MONET requires an altitude dimension named 'z'") from e

    paired_data = model_obj_for_pairing.monet.combine_point(
        obs.obj, radius_of_influence=mod.radius_of_influence, suffix=mod.label
    )
    if debug:
        print('After pairing: ', paired_data)

    p = pair()
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.filename = '{}_{}.nc'.format(p.obs, p.model)
    p.obj = paired_data.monet._df_to_da()
    p.obj = p.fix_paired_xarray(dset=p.obj)
    return p


def _pair_aircraft(model_obj, mod, obs, keys, obs_vars, mod_vars):
    """Pair aircraft observations.
    """
    if not isinstance(obs.obj, pd.DataFrame):
        obs.obj = obs.obj.to_dataframe()

    obs.obj = obs.obj.reset_index().dropna(
        subset=['pressure_obs', 'latitude', 'longitude']
    ).set_index('time')

    new_ds_obs = obs.obj.rename_axis('time_obs').reset_index().monet._df_to_da().set_coords(
        ['time_obs', 'pressure_obs']
    )

    ds_model = mod.util.combinetool.combine_da_to_da(model_obj, new_ds_obs, merge=False)
    ds_model = ds_model.interp(time=ds_model.time_obs.squeeze())

    paired_data = vert_interp(ds_model, obs.obj, keys + mod_vars)
    print('After pairing: ', paired_data)

    p = pair()
    p.type = 'aircraft'
    p.radius_of_influence = None
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.filename = '{}_{}.nc'.format(p.obs, p.model)
    p.obj = paired_data.set_index('time').to_xarray().expand_dims('x').transpose('time', 'x')
    return p


def _pair_sonde(model_obj, mod, obs, keys, obs_vars, mod_vars, control_dict):
    """Pair sonde observations.
    """
    if not isinstance(obs.obj, pd.DataFrame):
        obs.obj = obs.obj.to_dataframe()

    obs.obj = obs.obj.reset_index().dropna(
        subset=['pressure_obs', 'latitude', 'longitude']
    ).set_index('time')

    import datetime
    plot_dict_sonde = control_dict['plots']
    for grp_sonde, grp_dict_sonde in plot_dict_sonde.items():
        plot_type_sonde = grp_dict_sonde['type']
        plot_sonde_type_list_all = [
            'vertical_single_date',
            'vertical_boxplot_os',
            'density_scatter_plot_os',
        ]
        if plot_type_sonde in plot_sonde_type_list_all:
            station_name_sonde = grp_dict_sonde['station_name']
            cds_sonde = grp_dict_sonde['compare_date_single']
            obs.obj = obs.obj.loc[obs.obj['station'] == station_name_sonde[0]]
            obs.obj = obs.obj.loc[
                datetime.datetime(
                    cds_sonde[0],
                    cds_sonde[1],
                    cds_sonde[2],
                    cds_sonde[3],
                    cds_sonde[4],
                    cds_sonde[5],
                )
            ]
            break

    new_ds_obs = obs.obj.rename_axis('time_obs').reset_index().monet._df_to_da().set_coords(
        ['time_obs', 'pressure_obs']
    )
    ds_model = mod.util.combinetool.combine_da_to_da(model_obj, new_ds_obs, merge=False)
    ds_model = ds_model.interp(time=ds_model.time_obs.squeeze())
    paired_data = vert_interp(ds_model, obs.obj, keys + mod_vars)
    print('In pair function, After pairing: ', paired_data)

    p = pair()
    p.type = 'sonde'
    p.radius_of_influence = None
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.filename = '{}_{}.nc'.format(p.obs, p.model)
    p.obj = paired_data.set_index('time').to_xarray().expand_dims('x').transpose('time', 'x')
    return p


def _pair_mobile_or_ground(model_obj, mod, obs, keys, obs_vars, mod_vars):
    """Pair mobile or ground observations.
    """
    if not isinstance(obs.obj, pd.DataFrame):
        obs.obj = obs.obj.to_dataframe()

    obs.obj = obs.obj.reset_index().dropna(subset=['latitude', 'longitude']).set_index('time')

    new_ds_obs = obs.obj.rename_axis('time_obs').reset_index().monet._df_to_da().set_coords(
        ['time_obs']
    )

    ds_model = mod.util.combinetool.combine_da_to_da(model_obj, new_ds_obs, merge=False)
    ds_model = ds_model.interp(time=ds_model.time_obs.squeeze())

    paired_data = mobile_and_ground_pair(ds_model, obs.obj, keys + mod_vars)
    print('After pairing: ', paired_data)

    p = pair()
    if obs.obs_type.lower() == 'mobile':
        p.type = 'mobile'
    elif obs.obs_type.lower() == 'ground':
        p.type = 'ground'
    p.radius_of_influence = None
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.filename = '{}_{}.nc'.format(p.obs, p.model)
    p.obj = paired_data.set_index('time').to_xarray().expand_dims('x').transpose('time', 'x')
    return p


def _pair_sat_swath_clm(
    model_obj, mod, obs, keys, obs_vars, start_time, end_time, pairing_kwargs
):
    """Pair satellite swath column observations.
    """
    pairing_kws = {'apply_ak': True, 'mod_to_overpass': False}
    for key in pairing_kwargs.get(obs.obs_type.lower(), {}):
        pairing_kws[key] = pairing_kwargs[obs.obs_type.lower()][key]
    if 'apply_ak' not in pairing_kwargs.get(obs.obs_type.lower(), {}):
        print(
            'WARNING: The satellite pairing option apply_ak is being set to True '
            'because it was not specified in the YAML. Pairing will fail if '
            'there is no AK available.'
        )

    if obs.sat_type == 'omps_nm':
        if 'time' in obs.obj.dims:
            obs.obj = obs.obj.sel(time=slice(start_time, end_time))
            obs.obj = obs.obj.swap_dims({'time': 'x'})
        if pairing_kws['apply_ak'] is True:
            model_obj_subset = model_obj[keys + ['pres_pa_mid', 'surfpres_pa']]
            paired_data = sutil.omps_nm_pairing_apriori(model_obj_subset, obs.obj, keys)
        else:
            model_obj_subset = model_obj[keys + ['dp_pa']]
            paired_data = sutil.omps_nm_pairing(model_obj_subset, obs.obj, keys)
        paired_data = paired_data.where(paired_data.ozone_column.notnull())

    elif obs.sat_type == 'tropomi_l2_no2':
        i_no2_varname = [
            i
            for i, x in enumerate(obs_vars)
            if x == 'nitrogendioxide_tropospheric_column'
        ]
        if len(i_no2_varname) > 1:
            print('The TROPOMI NO2 variable is matched to more than one model variable.')
            print('Pairing is being done for model variable: ' + keys[i_no2_varname[0]])
        no2_varname = keys[i_no2_varname[0]]

        if pairing_kws['mod_to_overpass']:
            print('sampling model to 13:30 local overpass time')
            overpass_datetime = pd.date_range(
                start_time.replace(hour=13, minute=30),
                end_time.replace(hour=13, minute=30),
                freq='D',
            )
            model_obj = sutil.mod_to_overpasstime(
                model_obj, overpass_datetime, partial_col=no2_varname
            )
            model_obj = model_obj.transpose('time', 'z', 'y', 'x', Ellipsis)
        else:
            print('Warning: The pairing_kwarg mod_to_overpass is False.')
            print('Pairing will proceed assuming that the model data is already at overpass time.')
            from ..util.tools import calc_partialcolumn
            model_obj[f'{no2_varname}_col'] = calc_partialcolumn(model_obj, var=no2_varname)

        if pairing_kws['apply_ak'] is True:
            paired_data = no2util.trp_interp_swatogrd_ak(
                obs.obj, model_obj, no2varname=no2_varname
            )
        else:
            paired_data = no2util.trp_interp_swatogrd(
                obs.obj, model_obj, no2varname=no2_varname
            )
        paired_data = paired_data.sel(time=slice(start_time.date(), end_time.date()))

    elif 'tempo_l2' in obs.sat_type:
        if obs.sat_type == 'tempo_l2_no2':
            sat_sp, sp, key = 'NO2', 'vertical_column_troposphere', 'tempo_l2_no2'
        elif obs.sat_type == 'tempo_l2_hcho':
            sat_sp, sp, key = 'HCHO', 'vertical_column', 'tempo_l2_hcho'
        else:
            raise KeyError(
                f" You asked for {obs.sat_type}. "
                + "Only NO2 and HCHO L2 data have been implemented"
            )

        mod_sp = [k_sp for k_sp, v in mod.mapping[key].items() if v == sp]
        regrid_method = obs.regrid_method if obs.regrid_method is not None else "bilinear"
        paired_data_atswath = tempo_sutil.regrid_and_apply_weights(
            obs.obj, model_obj, species=mod_sp, method=regrid_method, tempo_sp=sat_sp
        )
        paired_data = tempo_sutil.back_to_modgrid_multiscan(
            paired_data_atswath, model_obj, method=regrid_method
        )
        paired_data = paired_data.sel(time=slice(start_time, end_time))

    else:
        raise ValueError(f"Unknown satellite type: {obs.sat_type}")

    p = pair()
    p.type = obs.obs_type
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.obj = paired_data
    return p


def _pair_sat_grid_clm(
    model_obj, mod, obs, keys, obs_vars, start_time, end_time, pairing_kwargs
):
    """Pair satellite grid column observations.
    """
    pairing_kws = {'apply_ak': True, 'mod_to_overpass': False}
    for key in pairing_kwargs.get(obs.obs_type.lower(), {}):
        pairing_kws[key] = pairing_kwargs[obs.obs_type.lower()][key]
    if 'apply_ak' not in pairing_kwargs[obs.obs_type.lower()]:
        print(
            'WARNING: The satellite pairing option apply_ak is being set to True '
            'because it was not specified in the YAML. Pairing will fail if '
            'there is no AK available.'
        )

    if len(keys) > 1:
        print('Caution: More than 1 variable is included in mapping keys.')
        print('Pairing code is calculating a column for {}'.format(keys[0]))

    if obs.sat_type == 'omps_l3':
        obs_dat = obs.obj.sel(time=slice(start_time.date(), end_time.date()))
        mod_dat = model_obj.sel(time=slice(start_time.date(), end_time.date()))
        paired_obsgrid = sutil.omps_l3_daily_o3_pairing(mod_dat, obs_dat, keys[0])

    elif obs.sat_type == 'mopitt_l3':
        if pairing_kws['apply_ak']:
            model_obj_subset = model_obj[keys + ['pres_pa_mid']]
            if pairing_kws['mod_to_overpass']:
                print('sampling model to 10:30 local overpass time')
                overpass_datetime = pd.date_range(
                    start_time.replace(hour=10, minute=30),
                    end_time.replace(hour=10, minute=30),
                    freq='D',
                )
                model_obj_subset = sutil.mod_to_overpasstime(
                    model_obj_subset, overpass_datetime
                )
            obs_dat = obs.obj.sel(time=slice(start_time.date(), end_time.date()))
            model_obj_subset = model_obj_subset.sel(
                time=slice(start_time.date(), end_time.date())
            )
            paired_obsgrid = sutil.mopitt_l3_pairing(
                model_obj_subset, obs_dat, keys[0], global_model=mod.is_global
            )
        else:
            raise NotImplementedError(
                "Pairing without averaging kernel has not been enabled for this dataset"
            )
    else:
        raise ValueError(f"Unknown satellite type: {obs.sat_type}")

    p = pair()
    p.type = obs.obs_type
    p.obs = obs.label
    p.model = mod.label
    p.model_vars = keys
    p.obs_vars = obs_vars
    p.obj = paired_obsgrid
    return p
