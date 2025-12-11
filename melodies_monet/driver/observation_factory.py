# SPDX-License-Identifier: Apache-2.0
#
"""
Factory for creating and configuring observation objects.
"""
import os
from .observation import observation

class ObservationFactory:
    def __init__(self):
        pass

    def create_observation(self, obs_name, obs_config, control_dict=None):
        """
        Create and configure an observation object.

        Parameters
        ----------
        obs_name : str
            The label/name of the observation.
        obs_config : dict
            The configuration dictionary for this specific observation.
        control_dict : dict, optional
            The full control dictionary, needed for some obs settings.

        Returns
        -------
        observation
            The configured observation object.
        """
        o = observation()
        o.obs = obs_name
        o.label = obs_name
        o.obs_type = obs_config['obs_type']

        if 'data_proc' in obs_config:
            o.data_proc = obs_config['data_proc']

        o.file = os.path.expandvars(obs_config['filename'])

        if 'debug' in obs_config:
            o.debug = obs_config['debug']
        if 'variables' in obs_config:
            o.variable_dict = obs_config['variables']
        if 'variable_summing' in obs_config:
            o.variable_summing = obs_config['variable_summing']
        if 'resample' in obs_config:
            o.resample = obs_config['resample']
        if 'time_var' in obs_config:
            o.time_var = obs_config['time_var']
        if 'ground_coordinate' in obs_config:
            o.ground_coordinate = obs_config['ground_coordinate']
        if 'sat_type' in obs_config:
            o.sat_type = obs_config['sat_type']

        return o
