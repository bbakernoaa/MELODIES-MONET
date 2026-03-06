import monet as m
import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime

from melodies_monet.driver import model, observation, pair


class analysis:
    def __init__(self):
        self.control = "control.yaml"
        self.models = {}
        self.obs = {}
        self.paired = {}
        self.start_time = None
        self.end_time = None

    def read_control(self, control=None):
        import yaml
        if control is not None: self.control = control
        with open(self.control, "r") as stream:
            self.control_dict = yaml.safe_load(stream)
        self.start_time = pd.Timestamp(self.control_dict["analysis"]["start_time"])
        self.end_time = pd.Timestamp(self.control_dict["analysis"]["end_time"])
        self.output_dir = os.path.expandvars(self.control_dict["analysis"]["output_dir"])

    def open_models(self):
        for mod in self.control_dict.get("model", {}):
            m_obj = model()
            m_obj.label = mod
            m_obj.open_model_files(control_dict=self.control_dict)
            self.models[mod] = m_obj

    def open_obs(self):
        for obs in self.control_dict.get("obs", {}):
            o_obj = observation()
            o_obj.label = obs
            o_obj.obs_type = self.control_dict["obs"][obs]["obs_type"]
            o_obj.file = os.path.expandvars(self.control_dict["obs"][obs]["filename"])
            o_obj.open_obs()
            self.obs[obs] = o_obj

    def pair_data(self):
        """Purely Xarray-based pairing."""
        for mod_label, mod in self.models.items():
            for obs_label in mod.mapping.keys():
                obs = self.obs[obs_label]

                # Create pair object
                p = pair()
                p.obs = obs_label
                p.model = mod_label

                # Call NEW upstream pair function (model.obj is Xarray, obs.obj is Xarray)
                p.obj = p.pair_data(mod.obj, obs.obj, suffix=f"_{mod_label}", interp_time=True)

                self.paired[f"{obs_label}_{mod_label}"] = p

    def stats(self):
        from monet_stats import proc_stats
        for p_label, p in self.paired.items():
            # Example stat calculation
            res = proc_stats.calc(p.obj, stat='MB', obsvar='OZONE', modvar=f'OZONE_{p.model}')
            print(f"Stats for {p_label}: MB={res}")
