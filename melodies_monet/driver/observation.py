# SPDX-License-Identifier: Apache-2.0
#
import os
import xarray as xr
import numpy as np

import monetio as mio
from melodies_monet.util import driver_util


class observation:
    """The observation class.

    A class with information and data from an observational dataset.
    """

    def __init__(self):
        """Initialize an :class:`observation` object."""
        self.obs = None
        self.label = None
        self.file = None
        self.obj = None
        """The data object (:class:`pandas.DataFrame` or :class:`xarray.Dataset`)."""
        self.type = "pt_src"
        self.sat_type = None
        self.sat_method = None
        self.data_proc = None
        self.variable_dict = None
        self.variable_summing = None
        self.resample = None
        self.time_var = None
        self.regrid_method = None

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    obs={self.obs!r},\n"
            f"    label={self.label!r},\n"
            f"    file={self.file!r},\n"
            f"    obj={repr(self.obj) if self.obj is None else '...'},\n"
            f"    type={self.type!r},\n"
            f"    sat_type={self.sat_type!r},\n"
            f"    sat_method={self.sat_method!r},\n"
            f"    data_proc={self.data_proc!r},\n"
            f"    variable_dict={self.variable_dict!r},\n"
            f"    resample={self.resample!r},\n"
            f"    time_var={self.time_var!r},\n"
            f"    regrid_method={self.regrid_method!r},\n"
            ")"
        )

    def open_obs(self, time_interval=None, control_dict=None):
        """Open the observational data using the updated Reader API."""
        from glob import glob
        from numpy import sort
        from melodies_monet import tutorial

        if self.file.startswith("example:"):
            example_id = ":".join(s.strip() for s in self.file.split(":")[1:])
            files = [tutorial.fetch_example(example_id)]
        else:
            files = sort(glob(self.file))

        assert len(files) >= 1, "need at least one"

        # Map MM obs type to monetio source names
        source_map = {
             "airnow": "airnow",
             "aqs": "aqs",
             "improve": "improve",
             "ish": "ish",
             "ish_lite": "ish_lite",
             "aeronet": "aeronet",
             "nadp": "nadp",
             "openaq": "openaq",
             "cems": "cems",
             "pams": "pams",
             "icartt": "icartt",
             "tolnet": "tolnet",
             "geoms": "geoms",
             "gml_ozonesonde": "gml_ozonesonde",
        }

        # Determine source
        obs_type_lower = self.obs_type.lower()
        source = source_map.get(obs_type_lower)

        # Check extensions for certain types
        _, extension = os.path.splitext(files[0])

        if source is not None:
             print(f"**** Reading {source} observation using updated Reader API...")
             # Handle ICARTT legacy constraint if needed, but updated API should handle it
             self.obj = mio.load(source, files=files)
        elif extension in [".ict", ".icartt"]:
             print("**** Reading ICARTT observation using updated Reader API...")
             self.obj = mio.load("icartt", files=files[0])
        elif obs_type_lower == "pt_sfc":
             # Fallback for generic point surface, try airnow as it was the previous default-ish behavior
             print("**** Reading pt_sfc observation (falling back to AirNow) using updated Reader API...")
             self.obj = mio.load("airnow", files=files)
        elif extension in [".csv"]:
             from melodies_monet.util.read_util import read_aircraft_obs_csv
             assert len(files) == 1, "MELODIES-MONET can only read one csv file"
             self.obj = read_aircraft_obs_csv(filename=files[0], time_var=self.time_var)
        elif extension in {".nc", ".ncf", ".netcdf", ".nc4"}:
             if len(files) > 1:
                 self.obj = xr.open_mfdataset(files)
             else:
                 self.obj = xr.open_dataset(files[0])
        else:
             raise ValueError(f"Unsupported observation type or extension: {self.obs_type}, {extension}")

        self.add_coordinates_ground()  # If ground site then add coordinates based on yaml if necessary
        self.mask_and_scale()  # mask and scale values from the control values
        self.rename_vars()  # rename any variables as necessary
        self.sum_variables()
        self.resample_data()
        self.filter_obs()

    def add_coordinates_ground(self):
        """Add latitude and longitude coordinates to data when the observation type is ground and
        ground_coordinate is specified

        Returns
        -------
        None
        """

        # If ground site
        if self.obs_type == "ground":
            if self.ground_coordinate and isinstance(self.ground_coordinate, dict):
                self.obj["latitude"] = xr.ones_like(self.obj["time"], dtype=np.float64) * self.ground_coordinate["latitude"]
                self.obj["longitude"] = xr.ones_like(self.obj["time"], dtype=np.float64) * self.ground_coordinate["longitude"]
            elif self.ground_coordinate and ~isinstance(self.ground_coordinate, dict):
                raise TypeError("The ground_coordinate option must be specified as a dict with keys latitude and longitude.")

    def rename_vars(self):
        """Rename any variables in observation with rename set."""
        self.obj = driver_util.apply_variable_rename(self.obj, self.variable_dict)

    def open_sat_obs(self, time_interval=None, control_dict=None):
        """Methods to opens satellite data observations using the updated Reader API."""
        from melodies_monet.util import time_interval_subset as tsub

        load_kwargs = self.variable_dict.copy() if self.variable_dict is not None else {}
        load_kwargs.update({"debug": self.debug})

        # Unified Reader API (monetio.load)
        supported_sources = [
            "goes", "nesdis_edr_viirs", "nesdis_eps_viirs", "modis_ornl",
            "nasa_modis", "nesdis_frp", "omps_l3", "omps_nm", "mopitt_l3",
            "tropomi_l2_no2", "tempo_l2_no2", "modis_l2"
        ]

        source = self.sat_type
        # Handle TEMPO naming
        if source == "tempo_l2":
             source = "tempo_l2_no2" # Default to no2 for now if ambiguous

        if source in supported_sources:
            print(f"**** Reading {source} satellite observation using updated Reader API...")

            # Subsetting logic for certain satellites
            files_to_load = self.file
            if source == "omps_nm" and time_interval is not None:
                files_to_load = tsub.subset_OMPS_l2(self.file, time_interval)
            elif source == "mopitt_l3" and time_interval is not None:
                files_to_load = tsub.subset_mopitt_l3(self.file, time_interval)
                load_kwargs.update({"varnames": ["column", "pressure_surf", "apriori_col", "apriori_surf", "apriori_prof", "ak_col"]})
            elif source == "modis_l2": # mapped to nasa_modis or modis_ornl? MM used _modis_l2_mm which is nasa_modis
                source = "nasa_modis"
                files_to_load = tsub.subset_MODIS_l2(self.file, time_interval)

            self.obj = mio.load(source, files=files_to_load, **load_kwargs)

            if source == "omps_nm":
                self.obj = self.obj.swap_dims({"x": "time"}).sortby("time")
                if time_interval is not None:
                    self.obj = self.obj.sel(time=slice(time_interval[0], time_interval[-1]))
            elif source == "mopitt_l3":
                from glob import glob
                if any(mtype in glob(self.file)[0] for mtype in ("MOP03JM", "MOP03NM", "MOP03TM")):
                    self.obj.attrs["monthly"] = True
                else:
                    self.obj.attrs["monthly"] = False
        else:
             print(f"**** Warning: {source} satellite reader not explicitly implemented in Unified API fallback may fail.")
             # Fallback to legacy if available in monetio.sat but not registered in load()
             # This is a bit of a placeholder as we want to encourage registration
             raise NotImplementedError(f"Satellite source {source} not supported in updated API yet.")

    def filter_obs(self):
        """Filter observations based on filter_dict.

        Returns
        -------
        None
        """
        if self.data_proc is not None:
            if "filter_dict" in self.data_proc:
                filter_dict = self.data_proc["filter_dict"]
                for column in filter_dict.keys():
                    filter_vals = filter_dict[column]["value"]
                    filter_op = filter_dict[column]["oper"]
                    if filter_op == "isin":
                        self.obj = self.obj.where(self.obj[column].isin(filter_vals), drop=True)
                    elif filter_op == "isnotin":
                        self.obj = self.obj.where(~self.obj[column].isin(filter_vals), drop=True)
                    elif filter_op == "==":
                        self.obj = self.obj.where(self.obj[column] == filter_vals, drop=True)
                    elif filter_op == ">":
                        self.obj = self.obj.where(self.obj[column] > filter_vals, drop=True)
                    elif filter_op == "<":
                        self.obj = self.obj.where(self.obj[column] < filter_vals, drop=True)
                    elif filter_op == ">=":
                        self.obj = self.obj.where(self.obj[column] >= filter_vals, drop=True)
                    elif filter_op == "<=":
                        self.obj = self.obj.where(self.obj[column] <= filter_vals, drop=True)
                    elif filter_op == "!=":
                        self.obj = self.obj.where(self.obj[column] != filter_vals, drop=True)
                    else:
                        raise ValueError(f"Filter operation {filter_op!r} is not supported")

    def mask_and_scale(self):
        """Mask and scale observations, including unit conversions and setting
        detection limits.
        """
        self.obj = driver_util.apply_mask_and_scale(self.obj, self.variable_dict)

    def sum_variables(self):
        """Sum any variables noted that should be summed to create new variables.
        This occurs after any unit scaling.
        """
        self.obj = driver_util.apply_variable_summing(self.obj, self.variable_summing, self.variable_dict)

    def resample_data(self):
        """Resample the obs df based on the value set in the control file.

        Returns
        -------
        None
        """

        ##Resample the data
        if self.resample is not None:
            self.obj = self.obj.resample(time=self.resample).mean(dim="time")

    def obs_to_df(self):
        """Convert and reformat observation object (:attr:`obj`) to dataframe.

        Returns
        -------
        None
        """
        try:
            self.obj = self.obj.to_dataframe().reset_index().drop(["x", "y"], axis=1)
        except KeyError:
            self.obj = self.obj.to_dataframe().reset_index().drop(["x"], axis=1)
