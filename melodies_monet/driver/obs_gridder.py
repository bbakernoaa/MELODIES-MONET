# SPDX-License-Identifier: Apache-2.0
#
"""
The observation gridder class.
"""
import numpy as np
import pandas as pd
import xarray as xr
from ..util import grid_util


class ObsGridder:
    """
    Class for handling observation gridding.
    """

    def __init__(self, analysis):
        """
        Initialize the ObsGridder.

        Parameters
        ----------
        analysis : analysis
            The analysis object containing the control dictionary and observation data.
        """
        self.analysis = analysis
        self.control_dict = analysis.control_dict

    def setup_obs_grid(self):
        """
        Setup a uniform observation grid.
        """
        ntime = self.control_dict['obs_grid']['ntime']
        nlat = self.control_dict['obs_grid']['nlat']
        nlon = self.control_dict['obs_grid']['nlon']
        self.analysis.obs_grid, self.analysis.obs_edges = grid_util.generate_uniform_grid(
            self.control_dict['obs_grid']['start_time'],
            self.control_dict['obs_grid']['end_time'],
            ntime, nlat, nlon)

        self.analysis.da_obs_grid = xr.DataArray(dims=['lon', 'lat'],
            coords={'lon': self.analysis.obs_grid['longitude'],
                    'lat': self.analysis.obs_grid['latitude']})

        for obs in self.control_dict['obs']:
            for var in self.control_dict['obs'][obs]['variables']:
                print('initializing gridded data and counts ', obs, var)
                self.analysis.obs_gridded_data[obs + '_' + var] = np.zeros([ntime, nlon, nlat], dtype=np.float32)
                self.analysis.obs_gridded_count[obs + '_' + var] = np.zeros([ntime, nlon, nlat], dtype=np.int32)

    def update_obs_gridded_data(self):
        """
        Update observation grid cell values and counts,
        for all observation datasets and parameters.
        """
        for obs in self.analysis.obs:
            for obs_time in self.analysis.obs[obs].obj:
                print('updating obs time: ', obs, obs_time)
                obs_timestamp = pd.to_datetime(
                    obs_time, format='%Y%j%H%M').timestamp()
                # print(obs_timestamp)
                for var in self.analysis.obs[obs].obj[obs_time]:
                    key = obs + '_' + var
                    print(key)
                    n_obs = self.analysis.obs[obs].obj[obs_time][var].size
                    grid_util.update_data_grid(
                        self.analysis.obs_edges['time_edges'],
                        self.analysis.obs_edges['lon_edges'],
                        self.analysis.obs_edges['lat_edges'],
                        np.full(n_obs, obs_timestamp, dtype=np.float32),
                        self.analysis.obs[obs].obj[obs_time].coords['lon'].values.flatten(),
                        self.analysis.obs[obs].obj[obs_time].coords['lat'].values.flatten(),
                        self.analysis.obs[obs].obj[obs_time][var].values.flatten(),
                        self.analysis.obs_gridded_count[key],
                        self.analysis.obs_gridded_data[key])

    def normalize_obs_gridded_data(self):
        """
        Normalize observation grid cell values where counts is not zero.
        Create data arrays for the obs_gridded_dataset dictionary.
        """
        self.analysis.obs_gridded_dataset = xr.Dataset()

        for obs in self.analysis.obs:
            for var in self.control_dict['obs'][obs]['variables']:
                key = obs + '_' + var
                print(key)
                grid_util.normalize_data_grid(
                    self.analysis.obs_gridded_count[key],
                    self.analysis.obs_gridded_data[key])
                da_data = xr.DataArray(
                    self.analysis.obs_gridded_data[key],
                    dims=['time', 'lon', 'lat'],
                    coords={'time': self.analysis.obs_grid['time'],
                            'lon': self.analysis.obs_grid['longitude'],
                            'lat': self.analysis.obs_grid['latitude']})
                da_count = xr.DataArray(
                    self.analysis.obs_gridded_count[key],
                    dims=['time', 'lon', 'lat'],
                    coords={'time': self.analysis.obs_grid['time'],
                            'lon': self.analysis.obs_grid['longitude'],
                            'lat': self.analysis.obs_grid['latitude']})
                self.analysis.obs_gridded_dataset[key + '_data'] = da_data
                self.analysis.obs_gridded_dataset[key + '_count'] = da_count
