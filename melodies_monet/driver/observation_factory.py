# SPDX-License-Identifier: Apache-2.0
#
"""
The observation factory class.
"""
import os
from .observation import observation


class ObservationFactory:
    """
    Factory class for creating observation objects.
    """

    def create_observation(self, name, config, time_interval=None, load_files=True, control_dict=None):
        """
        Create and configure an observation object.

        Parameters
        ----------
        name : str
            The name/label of the observation.
        config : dict
            The configuration dictionary for the specific observation.
        time_interval : list of pandas.Timestamp, optional
            Time interval to restrict the observation data.
        load_files : bool, optional
            Whether to open the observation files. Default is True.
        control_dict : dict, optional
            The full control dictionary, needed for some operations.

        Returns
        -------
        observation
            The configured observation object.
        """
        o = observation()
        o.obs = name
        o.label = name
        o.obs_type = config['obs_type']

        if 'data_proc' in config:
            o.data_proc = config['data_proc']

        o.file = os.path.expandvars(config['filename'])

        if 'debug' in config:
            o.debug = config['debug']
        if 'variables' in config:
            o.variable_dict = config['variables']
        if 'variable_summing' in config:
            o.variable_summing = config['variable_summing']
        if 'resample' in config:
            o.resample = config['resample']
        if 'time_var' in config:
            o.time_var = config['time_var']
        if 'ground_coordinate' in config:
            o.ground_coordinate = config['ground_coordinate']
        if 'sat_type' in config:
            o.sat_type = config['sat_type']

        if load_files:
            if o.obs_type in ['sat_swath_sfc', 'sat_swath_clm', 'sat_grid_sfc',
                                'sat_grid_clm', 'sat_swath_prof']:
                o.open_sat_obs(time_interval=time_interval, control_dict=control_dict)
            else:
                o.open_obs(time_interval=time_interval, control_dict=control_dict)

        return o
