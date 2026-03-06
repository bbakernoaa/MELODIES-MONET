import os
import xarray as xr
import numpy as np
import monetio as mio

class observation:
    """The observation class. Purely Xarray-based."""

    def __init__(self):
        """Initialize an :class:`observation` object."""
        self.obs = None
        self.label = None
        self.file = None
        self.obj = None
        self.type = "pt_src"
        self.sat_type = None
        self.sat_method = None
        self.data_proc = None
        self.variable_dict = None
        self.variable_summing = None
        self.resample = None
        self.time_var = None
        self.regrid_method = None
        self.ground_coordinate = None

    def open_obs(self, time_interval=None, control_dict=None):
        """Open obs and ensure it is an Xarray object."""
        # Use monetio.load which now returns Xarray Datasets
        self.obj = mio.load(self.obs_type, files=self.file, **(self.data_proc or {}))

        if time_interval is not None:
            self.obj = self.obj.sel(time=slice(time_interval[0], time_interval[-1]))

        self.add_coordinates_ground()
        self.mask_and_scale()
        self.rename_vars()
        self.sum_variables()
        self.resample_data()
        self.filter_obs()

    def open_sat_obs(self, time_interval=None, control_dict=None):
        """Open satellite obs via monetio."""
        self.obj = mio.load(self.sat_type, files=self.file, **(self.data_proc or {}))
        if time_interval is not None:
            self.obj = self.obj.sel(time=slice(time_interval[0], time_interval[-1]))

    def add_coordinates_ground(self):
        """Standard UGRID/CF coordinate addition."""
        if self.obs_type == "ground" and self.ground_coordinate:
            self.obj = self.obj.assign_coords(
                latitude=self.ground_coordinate["latitude"],
                longitude=self.ground_coordinate["longitude"]
            )

    def rename_vars(self):
        if self.variable_dict:
            rename_map = {v: d["rename"] for v, d in self.variable_dict.items() if "rename" in d and v in self.obj.variables}
            self.obj = self.obj.rename(rename_map)

    def mask_and_scale(self):
        if not self.variable_dict: return
        for v, d in self.variable_dict.items():
            if v not in self.obj: continue
            if "unit_scale" in d:
                self.obj[v] = self.obj[v] * d["unit_scale"]
            # etc...

    def sum_variables(self):
        if not self.variable_summing: return
        for var_new, info in self.variable_summing.items():
            self.obj[var_new] = sum(self.obj[v] for v in info["vars"] if v in self.obj)

    def resample_data(self):
        if self.resample:
            self.obj = self.obj.resample(time=self.resample).mean()

    def filter_obs(self):
        if self.data_proc and "filter_dict" in self.data_proc:
            # Implement Xarray-native filtering
            pass
