# SPDX-License-Identifier: Apache-2.0
#
import monet as m
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair

class analysis:
    """The analysis class. Conducts the execution flow of the driver."""

    def __init__(self):
        self.control = "control.yaml"
        self.control_dict = None
        self.models = {}
        self.obs = {}
        self.paired = {}
        self.start_time = None
        self.end_time = None
        self.time_intervals = None
        self.download_maps = True
        self.output_dir = None
        self.output_dir_save = None
        self.output_dir_read = None
        self.debug = False
        self.save = None
        self.read = None
        self.regrid = False
        self.target_grid = None
        self.obs_regridders = None
        self.model_regridders = None
        self.obs_grid = None
        self.obs_edges = None
        self.obs_gridded_data = {}
        self.obs_gridded_count = {}
        self.obs_gridded_dataset = None
        self.add_logo = True
        self.pairing_kwargs = {}

    def read_control(self, control=None):
        import yaml
        if control is not None: self.control = control
        with open(self.control, "r") as stream:
            self.control_dict = yaml.safe_load(stream)

        # Basic settings
        self.start_time = pd.Timestamp(self.control_dict["analysis"]["start_time"])
        self.end_time = pd.Timestamp(self.control_dict["analysis"]["end_time"])
        self.output_dir = os.path.expandvars(self.control_dict["analysis"]["output_dir"])
        self.output_dir_save = os.path.expandvars(self.control_dict["analysis"].get("output_dir_save", self.output_dir))
        self.output_dir_read = os.path.expandvars(self.control_dict["analysis"].get("output_dir_read", self.output_dir))
        self.debug = self.control_dict["analysis"].get("debug", False)
        self.save = self.control_dict["analysis"].get("save")
        self.read = self.control_dict["analysis"].get("read")
        self.add_logo = self.control_dict["analysis"].get("add_logo", True)
        self.regrid = self.control_dict["analysis"].get("regrid", False)
        self.target_grid = self.control_dict["analysis"].get("target_grid")
        self.pairing_kwargs = self.control_dict["analysis"].get("pairing_kwargs", {})

        # Time intervals for chunking
        if "time_interval" in self.control_dict["analysis"]:
            time_stamps = pd.date_range(start=self.start_time, end=self.end_time,
                                       freq=self.control_dict["analysis"]["time_interval"])
            if time_stamps[-1] < pd.Timestamp(self.end_time):
                time_stamps = time_stamps.append(pd.DatetimeIndex([self.end_time]))
            self.time_intervals = [[time_stamps[n], time_stamps[n + 1]] for n in range(len(time_stamps) - 1)]

    def open_models(self, time_interval=None, load_files=True):
        if "model" in self.control_dict:
            for mod_label in self.control_dict["model"]:
                cfg = self.control_dict["model"][mod_label]
                m_inst = Data(data_type="model")
                m_inst.label = mod_label
                m_inst.source = cfg["mod_type"]
                m_inst.file_str = cfg["files"]
                m_inst.mapping = cfg.get("mapping")
                m_inst.variable_dict = cfg.get("variables")
                m_inst.variable_summing = cfg.get("variable_summing")
                m_inst.radius_of_influence = cfg.get("radius_of_influence", 1e6)
                m_inst.is_global = cfg.get("is_global", False)
                m_inst.file_vert_str = cfg.get("files_vert")
                m_inst.file_surf_str = cfg.get("files_surf")
                m_inst.file_pm25_str = cfg.get("files_pm25")
                m_inst.mod_kwargs = cfg.get("mod_kwargs", {})
                m_inst.plot_kwargs = cfg.get("plot_kwargs")
                m_inst.scrip_file = cfg.get("scrip_file")

                if load_files:
                    m_inst.load(time_interval=time_interval)
                self.models[mod_label] = m_inst

    def open_obs(self, time_interval=None, load_files=True):
        if "obs" in self.control_dict:
            for obs_label in self.control_dict["obs"]:
                cfg = self.control_dict["obs"][obs_label]
                o_inst = Data(data_type="obs")
                o_inst.label = obs_label
                o_inst.obs_type = cfg["obs_type"]
                o_inst.file_str = cfg["filename"]
                o_inst.variable_dict = cfg.get("variables")
                o_inst.variable_summing = cfg.get("variable_summing")
                o_inst.resample = cfg.get("resample")
                o_inst.time_var = cfg.get("time_var")
                o_inst.ground_coordinate = cfg.get("ground_coordinate")
                o_inst.sat_type = cfg.get("sat_type")
                o_inst.data_proc = cfg.get("data_proc")
                o_inst.regrid_method = cfg.get("regrid_method")

                if load_files:
                    o_inst.load(time_interval=time_interval)
                self.obs[obs_label] = o_inst

    def pair_data(self):
        """Unified pairing logic leveraging monet.pair."""
        for mod_label, mod in self.models.items():
            for obs_to_pair in mod.mapping.keys():
                obs = self.obs[obs_to_pair]

                # Mapping and subsetting
                keys = list(mod.mapping[obs_to_pair].keys())
                obs_vars = list(mod.mapping[obs_to_pair].values())
                mod_vars = list(mod.variable_dict.keys()) if mod.variable_dict else []

                # Model variables for this pairing
                model_obj = mod.obj[list(set(keys + mod_vars))]

                # Perform pairing
                paired_data = m.pair(
                    model_obj, obs.obj,
                    radius_of_influence=mod.radius_of_influence,
                    suffix=mod.label,
                    type=obs.obs_type.lower(),
                    **self.pairing_kwargs.get(obs.obs_type.lower(), {})
                )

                p_inst = pair()
                p_inst.obs = obs.label
                p_inst.model = mod.label
                p_inst.model_vars = keys
                p_inst.obs_vars = obs_vars
                p_inst.type = obs.obs_type.lower()
                p_inst.obj = paired_data

                label = f"{p_inst.obs}_{p_inst.model}"
                self.paired[label] = p_inst

    def save_analysis(self):
        if self.save:
            for attr, cfg in self.save.items():
                obj = getattr(self, attr)
                if cfg["method"] == "netcdf":
                    from melodies_monet.util.write_util import write_analysis_ncf
                    # Use provided prefix if available
                    write_analysis_ncf(obj=obj, output_dir=self.output_dir_save,
                                     fn_prefix=cfg.get("prefix", ""),
                                     keep_groups=cfg.get("data", "all"))
                elif cfg["method"] == "pkl":
                    from melodies_monet.util.write_util import write_pkl
                    write_pkl(obj=obj, output_name=os.path.join(self.output_dir_save, cfg["output_name"]))

    # Add other placeholder methods for grid/plots/stats if needed...
    def stats(self): pass
    def plotting(self): pass
