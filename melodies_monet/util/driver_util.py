# SPDX-License-Identifier: Apache-2.0
"""
Utilities for driver logic, including masking, scaling, renaming, and summing variables.
Ensures backend-agnosticism and tracks data lineage.
"""

import datetime
import xarray as xr
import pandas as pd
from typing import Union, Dict, Any

def _update_history(obj: Union[xr.Dataset, xr.DataArray], message: str):
    """Update the history attribute of an xarray object."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history = obj.attrs.get("history", "")
    new_history = f"{now}: {message}\n{history}"
    obj.attrs["history"] = new_history.strip()

def apply_mask_and_scale(obj: xr.Dataset, variable_dict: Dict[str, Any]) -> xr.Dataset:
    """
    Apply masking (min, max, nan values) and unit scaling to variables.

    Parameters
    ----------
    obj : xarray.Dataset
        The data to process.
    variable_dict : dict
        Configuration dictionary for variables.

    Returns
    -------
    xarray.Dataset
        Processed data.
    """
    if variable_dict is None:
        return obj

    if isinstance(obj, xr.Dataset):
        for v in obj.data_vars:
            if v in variable_dict:
                d = variable_dict[v]
                # Observation specific masking
                if "obs_min" in d:
                    obj[v] = obj[v].where(obj[v] >= d["obs_min"])
                if "obs_max" in d:
                    obj[v] = obj[v].where(obj[v] <= d["obs_max"])
                if "nan_value" in d:
                    obj[v] = obj[v].where(obj[v] != d["nan_value"])

                # Scaling
                scale = d.get("unit_scale", 1)
                method = d.get("unit_scale_method")
                if method == "*":
                    obj[v] = obj[v] * scale
                elif method == "/":
                    obj[v] = obj[v] / scale
                elif method == "+":
                    obj[v] = obj[v] + scale
                elif method == "-":
                    obj[v] = obj[v] - scale

                # LLOD masking
                if "LLOD_value" in d:
                    obj[v] = obj[v].where(obj[v] != d["LLOD_value"], d.get("LLOD_setvalue"))

        _update_history(obj, "Applied masking and scaling based on variable_dict")

    return obj

def apply_variable_rename(obj: xr.Dataset, variable_dict: Dict[str, Any]) -> xr.Dataset:
    """
    Rename variables in a dataset based on the configuration.

    Parameters
    ----------
    obj : xarray.Dataset
        The data to process.
    variable_dict : dict
        Configuration dictionary for variables.

    Returns
    -------
    xarray.Dataset
        Processed data.
    """
    if variable_dict is None:
        return obj

    rename_dict = {}
    if isinstance(obj, xr.Dataset):
        # Coordinates might also need renaming (important for obs)
        all_vars = list(obj.data_vars) + list(obj.coords)
        for v in all_vars:
            if v in variable_dict:
                d = variable_dict[v]
                if "rename" in d:
                    new_name = d["rename"]
                    rename_dict[v] = new_name
                    # Also update the variable_dict key to match the new name for subsequent steps
                    # Note: this modifies the dict in place or requires care.
                    # In MM, it was updating self.variable_dict[new_name] = self.variable_dict.pop(v)

        if rename_dict:
            obj = obj.rename(rename_dict)
            _update_history(obj, f"Renamed variables: {rename_dict}")

            # We must also update the variable_dict if we want subsequent steps to find them
            for old_v, new_v in rename_dict.items():
                if old_v in variable_dict:
                    variable_dict[new_v] = variable_dict.pop(old_v)

    return obj

def apply_variable_summing(obj: xr.Dataset, variable_summing: Dict[str, Any], variable_dict: Dict[str, Any] = None) -> xr.Dataset:
    """
    Sum variables to create new variables.

    Parameters
    ----------
    obj : xr.Dataset
        The data to process.
    variable_summing : dict
        Configuration for summing variables.
    variable_dict : dict, optional
        Variable configuration dictionary to update with new variables.

    Returns
    -------
    xr.Dataset
        Processed data.
    """
    if variable_summing is None:
        return obj

    for var_new, info in variable_summing.items():
        if var_new in obj.variables:
            continue # Already exists, or should we raise error? Legacy MM raised error.

        vars_to_sum = info.get("vars", [])
        if not vars_to_sum:
            continue

        # Use vectorized xarray sum
        # We start with the first variable and add the rest
        # This preserves laziness if inputs are dask-backed
        for i, v in enumerate(vars_to_sum):
            if i == 0:
                obj[var_new] = obj[v].copy()
            else:
                obj[var_new] = obj[var_new] + obj[v]

        if variable_dict is not None:
            variable_dict[var_new] = info

    _update_history(obj, f"Created summed variables: {list(variable_summing.keys())}")
    return obj
