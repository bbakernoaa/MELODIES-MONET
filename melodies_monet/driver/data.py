# SPDX-License-Identifier: Apache-2.0
#
import os

import numpy as np
import xarray as xr

import monetio as mio
from melodies_monet import tutorial
from melodies_monet.util import time_interval_subset as tsub


class Data:
    """The unified Data class for both models and observations.

    Acts as a conductor, leveraging monetio.load for ingestion and
    standardizing processing.
    """

    def __init__(self, data_type="model"):
        self.data_type = data_type
        self.label = None
        self.source = None
        self.file_str = None
        self.files = None
        self.obj = None

        # Metadata and processing
        self.variable_dict = None
        self.variable_summing = None
        self.mapping = None
        self.mod_kwargs = {}
        self.plot_kwargs = None

        # Specific to models
        self.is_global = False
        self.radius_of_influence = 1e6
        self.file_vert_str = None
        self.files_vert = None
        self.file_surf_str = None
        self.files_surf = None
        self.file_pm25_str = None
        self.files_pm25 = None
        self.scrip_file = None
        self.proj = None

        # Specific to observations
        self.obs_type = "pt_sfc"
        self.sat_type = None
        self.data_proc = None
        self.resample = None
        self.time_var = None
        self.ground_coordinate = None
        self.regrid_method = None
        self.use_dtn = False

    def __repr__(self):
        return f"Data(type={self.data_type!r}, label={self.label!r}, source={self.source!r})"

    def from_dict(self, cfg: dict):
        """
        Update attributes from a configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary for the data source.

        Returns
        -------
        Data
            The updated Data instance.
        """
        if self.data_type == "model":
            self.source = cfg.get("mod_type", cfg.get("source"))
            self.file_str = cfg.get("files", cfg.get("filename"))
            self.mapping = cfg.get("mapping")
            self.variable_dict = cfg.get("variables")
            self.variable_summing = cfg.get("variable_summing")
            self.radius_of_influence = cfg.get("radius_of_influence", 1e6)
            self.is_global = cfg.get("is_global", False)
            self.file_vert_str = cfg.get("files_vert")
            self.file_surf_str = cfg.get("files_surf")
            self.file_pm25_str = cfg.get("files_pm25")
            self.mod_kwargs = cfg.get("mod_kwargs", {})
            self.plot_kwargs = cfg.get("plot_kwargs")
            self.scrip_file = cfg.get("scrip_file")
        else:
            self.obs_type = cfg.get("obs_type", "pt_sfc")
            self.source = cfg.get("source")
            self.file_str = cfg.get("filename", cfg.get("files"))
            self.variable_dict = cfg.get("variables")
            self.variable_summing = cfg.get("variable_summing")
            self.resample = cfg.get("resample")
            self.time_var = cfg.get("time_var")
            self.ground_coordinate = cfg.get("ground_coordinate")
            self.sat_type = cfg.get("sat_type")
            self.data_proc = cfg.get("data_proc")
            self.regrid_method = cfg.get("regrid_method")

        self.use_dtn = cfg.get("use_dtn", False)
        return self

    def glob_files(self, time_interval: list = None) -> None:
        """
        Expand file patterns and optionally subset by time.

        Parameters
        ----------
        time_interval : list of pd.Timestamp, optional
            A list containing [start_time, end_time] to subset the data.

        Returns
        -------
        None
        """
        from glob import glob

        from numpy import sort

        if not self.file_str:
            return

        # Expand environment variables
        expanded_file_str = os.path.expandvars(self.file_str)

        if isinstance(self.file_str, list):
            self.files = sorted([os.path.expandvars(f) for f in self.file_str])
        elif expanded_file_str.startswith("example:"):
            example_id = ":".join(s.strip() for s in expanded_file_str.split(":")[1:])
            self.files = [tutorial.fetch_example(example_id)]
        elif expanded_file_str.lower().endswith(".txt"):
            with open(expanded_file_str, "r") as f:
                self.files = [os.path.expandvars(line.strip()) for line in f if line.strip()]
        else:
            self.files = sort(glob(expanded_file_str)).tolist()

        # Auxiliary files
        if self.file_vert_str:
            self.files_vert = sort(glob(os.path.expandvars(self.file_vert_str))).tolist()
        if self.file_surf_str:
            self.files_surf = sort(glob(os.path.expandvars(self.file_surf_str))).tolist()
        if self.file_pm25_str:
            self.files_pm25 = sort(glob(os.path.expandvars(self.file_pm25_str))).tolist()

        # Time subsetting
        if time_interval is not None:
            if self.source == "raqms":
                self.files = tsub.subset_model_filelist(self.files, "%m_%d_%Y_%HZ", "6H", time_interval)
            elif self.sat_type == "omps_nm":
                self.files = tsub.subset_OMPS_l2(self.files, time_interval)
            elif self.sat_type == "mopitt_l3":
                self.files = tsub.subset_mopitt_l3(self.files, time_interval)
            elif self.sat_type == "modis_l2":
                self.files = tsub.subset_MODIS_l2(self.files, time_interval)

    def load(self, time_interval: list = None) -> None:
        """
        Load data using monetio.load and apply standard processing.

        Parameters
        ----------
        time_interval : list of pd.Timestamp, optional
            A list containing [start_time, end_time] to subset the data.

        Returns
        -------
        None
        """
        import pandas as pd

        self.glob_files(time_interval=time_interval)

        if not self.files:
            print(f"WARNING: No files found for {self.label}")
            return

        load_kwargs = self.mod_kwargs.copy()

        if self.data_type == "model":
            list_input_var = self._get_model_var_list()
            load_kwargs.update({"var_list": list_input_var})
            if self.files_vert:
                load_kwargs.update({"fname_vert": self.files_vert})
            if self.files_surf:
                load_kwargs.update({"fname_surf": self.files_surf})
            if self.files_pm25:
                load_kwargs.update({"fname_pm25": self.files_pm25})
            if self.source == "cesm_se":
                scrip = os.path.expandvars(self.scrip_file) if self.scrip_file else ""
                if scrip.startswith("example:"):
                    scrip = tutorial.fetch_example(scrip.split(":", 1)[1].strip())
                load_kwargs.update({"scrip_file": scrip})

        try:
            source = self.source or self._guess_source()
            # Handle special readers or generic fallback
            self.obj = mio.load(source, files=self.files, **load_kwargs)
        except Exception as e:
            print(f"Error loading {self.label} ({self.source}): {e}. Falling back to generic xarray.")
            if len(self.files) > 1:
                self.obj = xr.open_mfdataset(self.files, **load_kwargs)
            else:
                self.obj = xr.open_dataset(self.files[0], **load_kwargs)

        if self.data_type == "obs":
            self.add_coordinates_ground()

        self.mask_and_scale()
        self.rename_vars()
        self.sum_variables()

        if self.data_type == "obs":
            self.resample_data()
            self.filter_obs()
            if time_interval is not None and "time" in self.obj.dims:
                self.obj = self.obj.sel(time=slice(time_interval[0], time_interval[-1]))

        # Scientific hygiene: Update history
        history = self.obj.attrs.get("history", "")
        new_history = f"{pd.Timestamp.now()}: Loaded and processed {self.label} data via MELODIES-MONET Orchestrator."
        self.obj.attrs["history"] = f"{new_history}\n{history}"

    def _get_model_var_list(self):
        """Gather all variables required for pairing and summing."""
        if not self.variable_dict and not self.mapping:
            return None
        vars_req = set()
        if self.variable_dict:
            vars_req.update(self.variable_dict.keys())
        if self.variable_summing:
            for v in self.variable_summing.values():
                vars_req.update(v["vars"])
        if self.mapping:
            for m in self.mapping.values():
                vars_req.update(m.keys())
        # Remove standardized names
        for vn in ["temperature_k", "pres_pa_mid"]:
            if vn in vars_req:
                vars_req.remove(vn)
        return list(vars_req)

    def _guess_source(self):
        fn = self.files[0]
        ext = os.path.splitext(fn)[1].lower()
        if ext in {".ict", ".icartt"}:
            return "icartt"
        if ext == ".csv":
            return "aircraft_csv"
        return self.label.lower()

    def add_coordinates_ground(self):
        if self.obs_type == "ground" and self.ground_coordinate:
            self.obj["latitude"] = xr.ones_like(self.obj["time"], dtype=float) * self.ground_coordinate["latitude"]
            self.obj["longitude"] = xr.ones_like(self.obj["time"], dtype=float) * self.ground_coordinate["longitude"]

    def rename_vars(self):
        if not self.variable_dict:
            return
        rename_map = {v: d["rename"] for v, d in self.variable_dict.items() if v in self.obj.variables and "rename" in d}
        if rename_map:
            self.obj = self.obj.rename(rename_map)
            for old, new in rename_map.items():
                self.variable_dict[new] = self.variable_dict.pop(old)

    def mask_and_scale(self):
        if not self.variable_dict:
            return
        for v in self.obj.data_vars:
            if v in self.variable_dict:
                d = self.variable_dict[v]
                # Removal of bad values (Lazy)
                if "obs_min" in d:
                    self.obj[v] = self.obj[v].where(self.obj[v] >= d["obs_min"])
                if "obs_max" in d:
                    self.obj[v] = self.obj[v].where(self.obj[v] <= d["obs_max"])
                if "nan_value" in d:
                    self.obj[v] = self.obj[v].where(self.obj[v] != d["nan_value"])

                # Scaling
                scale = d.get("unit_scale", 1.0)
                method = d.get("unit_scale_method", "*")
                if method == "*":
                    self.obj[v] = self.obj[v] * scale
                elif method == "/":
                    self.obj[v] = self.obj[v] / scale
                elif method == "+":
                    self.obj[v] = self.obj[v] + scale
                elif method == "-":
                    self.obj[v] = self.obj[v] - scale

                # Detection limits
                if "LLOD_value" in d:
                    self.obj[v] = self.obj[v].where(self.obj[v] != d["LLOD_value"], d.get("LLOD_setvalue", np.nan))

    def sum_variables(self):
        if not self.variable_summing:
            return
        for var_new, info in self.variable_summing.items():
            if var_new in self.obj.variables:
                continue
            self.obj[var_new] = sum(self.obj[v] for v in info["vars"] if v in self.obj.variables)
            if self.variable_dict is None:
                self.variable_dict = {}
            self.variable_dict[var_new] = info

    def resample_data(self):
        if self.resample:
            self.obj = self.obj.resample(time=self.resample).mean(dim="time")

    def filter_obs(self):
        if not self.data_proc or "filter_dict" not in self.data_proc:
            return
        fd = self.data_proc["filter_dict"]
        for col, cfg in fd.items():
            if col not in self.obj.variables:
                continue
            val = cfg["value"]
            op = cfg["oper"]
            if op == "isin":
                self.obj = self.obj.where(self.obj[col].isin(val), drop=True)
            elif op == "isnotin":
                self.obj = self.obj.where(~self.obj[col].isin(val), drop=True)
            elif op == "==":
                self.obj = self.obj.where(self.obj[col] == val, drop=True)
            elif op == "!=":
                self.obj = self.obj.where(self.obj[col] != val, drop=True)
            elif op == ">":
                self.obj = self.obj.where(self.obj[col] > val, drop=True)
            elif op == "<":
                self.obj = self.obj.where(self.obj[col] < val, drop=True)
            elif op == ">=":
                self.obj = self.obj.where(self.obj[col] >= val, drop=True)
            elif op == "<=":
                self.obj = self.obj.where(self.obj[col] <= val, drop=True)
