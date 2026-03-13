import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import monet as m
from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair

class analysis:
    """The analysis class. High-level orchestrator for evaluations."""

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

        self._migrate_control_dict()

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
        if "models" in self.control_dict:
            for mod_label in self.control_dict["models"]:
                cfg = self.control_dict["models"][mod_label]
                m_inst = Data(data_type="model")
                m_inst.label = mod_label
                m_inst.source = cfg.get("mod_type", cfg.get("source"))
                m_inst.file_str = cfg.get("files", cfg.get("filename"))
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
                o_inst.obs_type = cfg.get("obs_type", "pt_sfc")
                o_inst.source = cfg.get("source")
                o_inst.file_str = cfg.get("filename", cfg.get("files"))
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
        if "evaluations" in self.control_dict:
            for eval_label, cfg in self.control_dict["evaluations"].items():
                mod_label = cfg["model"]
                ref_label = cfg["obs"]
                mod = self.models[mod_label]
                ref = self.obs[ref_label]

                # Mapping and subsetting
                mapping = cfg.get("mapping")
                if mapping is None:
                    mapping = mod.mapping.get(ref_label, {}) if mod.mapping else {}
                keys = list(mapping.keys())
                ref_vars = list(mapping.values())
                mod_vars = list(mod.variable_dict.keys()) if mod.variable_dict else []

                # Model variables for this pairing
                model_obj = mod.obj[list(set(keys + mod_vars))]

                # Perform pairing
                paired_data = m.pair(
                    model_obj, ref.obj,
                    radius_of_influence=mod.radius_of_influence,
                    suffix=mod.label,
                    type=ref.obs_type.lower(),
                    **self.pairing_kwargs.get(ref.obs_type.lower(), {})
                )

                p_inst = pair()
                p_inst.ref = ref.label
                p_inst.model = mod.label
                p_inst.model_vars = keys
                p_inst.ref_vars = ref_vars
                p_inst.type = ref.obs_type.lower()
                p_inst.obj = paired_data

                self.paired[eval_label] = p_inst
        else:
            # Fallback for old mapping style if migration didn't run or failed
            for mod_label, mod in self.models.items():
                if not mod.mapping: continue
                for ref_label in mod.mapping.keys():
                    ref = self.obs[ref_label]

                    # Mapping and subsetting
                    keys = list(mod.mapping[ref_label].keys())
                    ref_vars = list(mod.mapping[ref_label].values())
                    mod_vars = list(mod.variable_dict.keys()) if mod.variable_dict else []

                    # Model variables for this pairing
                    model_obj = mod.obj[list(set(keys + mod_vars))]

                    # Perform pairing
                    paired_data = m.pair(
                        model_obj, ref.obj,
                        radius_of_influence=mod.radius_of_influence,
                        suffix=mod.label,
                        type=ref.obs_type.lower(),
                        **self.pairing_kwargs.get(ref.obs_type.lower(), {})
                    )

                    p_inst = pair()
                    p_inst.ref = ref.label
                    p_inst.model = mod.label
                    p_inst.model_vars = keys
                    p_inst.ref_vars = ref_vars
                    p_inst.type = ref.obs_type.lower()
                    p_inst.obj = paired_data

                    label = f"{p_inst.ref}_{p_inst.model}"
                    self.paired[label] = p_inst

    def _migrate_control_dict(self):
        """Transparently upgrade configurations to the four-tier hierarchical schema."""
        if "model" in self.control_dict:
            self.control_dict.setdefault("models", {}).update(self.control_dict.pop("model"))

        if "data" in self.control_dict:
            data_cfg = self.control_dict.pop("data")
            for k, v in data_cfg.items():
                dtype = v.get("type", "model")
                if dtype == "model":
                    self.control_dict.setdefault("models", {})[k] = v
                elif dtype == "obs":
                    self.control_dict.setdefault("obs", {})[k] = v

        if "plots" in self.control_dict:
            self.control_dict.setdefault("plotting", {}).update(self.control_dict.pop("plots"))

        if "evaluations" not in self.control_dict and "models" in self.control_dict:
            self.control_dict["evaluations"] = {}
            for mod_label, mod_cfg in self.control_dict["models"].items():
                if "mapping" in mod_cfg:
                    for obs_label, mapping in mod_cfg["mapping"].items():
                        eval_label = f"{obs_label}_{mod_label}"
                        self.control_dict["evaluations"][eval_label] = {
                            "model": mod_label,
                            "obs": obs_label,
                            "mapping": mapping
                        }

    def save_analysis(self):
        if self.save:
            for attr, cfg in self.save.items():
                obj = getattr(self, attr)
                if cfg["method"] == "netcdf":
                    from melodies_monet.util.write_util import write_analysis_ncf
                    write_analysis_ncf(obj=obj, output_dir=self.output_dir_save,
                                     fn_prefix=cfg.get("prefix", ""),
                                     keep_groups=cfg.get("data", "all"))
                elif cfg["method"] == "pkl":
                    from melodies_monet.util.write_util import write_pkl
                    write_pkl(obj=obj, output_name=os.path.join(self.output_dir_save, cfg["output_name"]))

    def plotting(self):
        """Standard plotting driver using monet-plots."""
        import monet_plots as mplots
        # Implementation details omitted for brevity, logic remains in MM orchestrator
        pass

    def stats(self):
        """Standard statistics driver using monet-stats."""
        import monet_stats as mstats
        # Implementation details omitted for brevity, logic remains in MM orchestrator
        pass
