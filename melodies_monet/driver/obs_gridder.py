# SPDX-License-Identifier: Apache-2.0
#
"""
Class for handling observation gridding.
"""
import pandas as pd
import numpy as np
import xarray as xr
from ..util import grid_util

class ObsGridder:
    def __init__(self, control_dict):
        """
        Initialize the ObsGridder.

        Parameters
        ----------
        control_dict : dict
            The control dictionary containing 'obs_grid' settings.
        """
        self.control_dict = control_dict
        self.obs_grid = None
        self.obs_edges = None
        self.da_obs_grid = None
        self.obs_gridded_data = {}
        self.obs_gridded_count = {}
        self.obs_gridded_dataset = None

    def setup_obs_grid(self):
        """
        Setup a uniform observation grid.
        """
        if 'obs_grid' not in self.control_dict:
             print("Warning: 'obs_grid' not found in control dictionary.")
             return

        ntime = self.control_dict['obs_grid']['ntime']
        nlat = self.control_dict['obs_grid']['nlat']
        nlon = self.control_dict['obs_grid']['nlon']
        self.obs_grid, self.obs_edges = grid_util.generate_uniform_grid(
            self.control_dict['obs_grid']['start_time'],
            self.control_dict['obs_grid']['end_time'],
            ntime, nlat, nlon)

        self.da_obs_grid = xr.DataArray(dims=['lon', 'lat'],
            coords={'lon': self.obs_grid['longitude'],
                    'lat': self.obs_grid['latitude']})
        # print(self.da_obs_grid)

        # Initialize arrays based on observations in control_dict
        if 'obs' in self.control_dict:
            for obs in self.control_dict['obs']:
                if 'variables' in self.control_dict['obs'][obs]:
                    for var in self.control_dict['obs'][obs]['variables']:
                        print('initializing gridded data and counts ', obs, var)
                        self.obs_gridded_data[obs + '_' + var] = np.zeros([ntime, nlon, nlat], dtype=np.float32)
                        self.obs_gridded_count[obs + '_' + var] = np.zeros([ntime, nlon, nlat], dtype=np.int32)

    def update_obs_gridded_data(self, observations):
        """
        Update observation grid cell values and counts,
        for all observation datasets and parameters.

        Parameters
        ----------
        observations : dict
            Dictionary of observation objects (e.g. analysis.obs).
        """
        if self.obs_edges is None:
            print("Warning: Observation grid not setup. Call setup_obs_grid() first.")
            return

        for obs_name in observations:
            obs_obj = observations[obs_name]
            for obs_time in obs_obj.obj:
                print('updating obs time: ', obs_name, obs_time)
                obs_timestamp = pd.to_datetime(
                    obs_time, format='%Y%j%H%M').timestamp()
                # print(obs_timestamp)
                for var in obs_obj.obj[obs_time]:
                    key = obs_name + '_' + var
                    print(key)
                    if key not in self.obs_gridded_data:
                         continue # Skip if not initialized (e.g. variable mismatch)

                    n_obs = obs_obj.obj[obs_time][var].size
                    grid_util.update_data_grid(
                        self.obs_edges['time_edges'],
                        self.obs_edges['lon_edges'],
                        self.obs_edges['lat_edges'],
                        np.full(n_obs, obs_timestamp, dtype=np.float32),
                        obs_obj.obj[obs_time].coords['lon'].values.flatten(),
                        obs_obj.obj[obs_time].coords['lat'].values.flatten(),
                        obs_obj.obj[obs_time][var].values.flatten(),
                        self.obs_gridded_count[key],
                        self.obs_gridded_data[key])

    def normalize_obs_gridded_data(self):
        """
        Normalize observation grid cell values where counts is not zero.
        Create data arrays for the obs_gridded_dataset dictionary.

        Returns
        -------
        xr.Dataset
            The normalized gridded dataset.
        """
        self.obs_gridded_dataset = xr.Dataset()

        if 'obs' not in self.control_dict:
            return self.obs_gridded_dataset

        for obs in self.control_dict['obs']:
            if 'variables' in self.control_dict['obs'][obs]:
                for var in self.control_dict['obs'][obs]['variables']:
                    key = obs + '_' + var
                    if key not in self.obs_gridded_data:
                        continue

                    print(key)
                    grid_util.normalize_data_grid(
                        self.obs_gridded_count[key],
                        self.obs_gridded_data[key])
                    da_data = xr.DataArray(
                        self.obs_gridded_data[key],
                        dims=['time', 'lon', 'lat'],
                        coords={'time': self.obs_grid['time'],
                                'lon': self.obs_grid['longitude'],
                                'lat': self.obs_grid['latitude']})
                    da_count = xr.DataArray(
                        self.obs_gridded_count[key],
                        dims=['time', 'lon', 'lat'],
                        coords={'time': self.obs_grid['time'],
                                'lon': self.obs_grid['longitude'],
                                'lat': self.obs_grid['latitude']})
                    self.obs_gridded_dataset[key + '_data'] = da_data
                    self.obs_gridded_dataset[key + '_count'] = da_count

        return self.obs_gridded_dataset
