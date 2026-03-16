# SPDX-License-Identifier: Apache-2.0
#
from __future__ import division

from builtins import range

import numpy as np
import xarray as xr

__author__ = "barry"


R = 8.31446261815324  # m3 * Pa / K / mol
N_A = 6.02214076e23


from monet.util.tools import (
    _force_forder,
    calc_24hr_ave,
    calc_3hr_ave,
    calc_8hr_rolling_max,
    calc_annual_ave,
    findclosest,
    get_epa_region_bounds,
    get_epa_region_df,
    get_giorgi_region_bounds,
    get_giorgi_region_df,
    get_relhum,
    kolmogorov_zurbenko_filter,
    linregress,
    long_to_wide,
    search_listinlist,
    wsdir2uv,
)


def list_contains(list1, list2):
    """Return True if any item in `list1` is also in `list2`."""
    for m in list1:
        for n in list2:
            if n == m:
                return True

    return False








def find_obs_time_bounds(files=[], **kwargs):
    """Function to read a series of files and print a list of min and max times for each.

    Parameters
    ----------
    files : str or iterable
        str or list of str containing filenames that should be read.
    **kwargs
        Additional arguments passed to monetio.load (e.g., source, time_var).

    Returns
    -------
    bounds : dict
        Dict containing time bounds for each file.

    """
    import os

    import monetio as mio

    if isinstance(files, str):
        files = [files]

    bounds = {}
    source_arg = kwargs.get("source")
    for file in files:
        _, extension = os.path.splitext(file)
        try:
            if source_arg is not None:
                obs = mio.load(source_arg, files=file, **kwargs)
            elif extension in {".nc", ".ncf", ".netcdf", ".nc4"}:
                obs = xr.open_dataset(file)
            elif extension in [".ict", ".icartt"]:
                obs = mio.load("icartt", files=file, **kwargs)
            else:
                raise ValueError(f"extension {extension!r} currently unsupported. Please specify 'source'.")
        except Exception as e:
            print("something happened opening file:", e)
            return

        time_min = obs["time"].min()
        time_max = obs["time"].max()

        print("For {}, time bounds are, Min: {}, Max: {}".format(file, time_min, time_max))
        bounds[file] = {"Min": time_min, "Max": time_max}

        del obs

    return bounds


def loop_pairing(control, file_pairs_yaml="", file_pairs={}, save_types=["paired"]):
    """Function to loop over sets of pairings and save them out as multiple netcdf files.

    Parameters
    ----------
    control : str
        str containing path to control file.

    file_pairs : dict (optional)
        Dict containing filenames for obs and models. This should be specified if file_pairs_yaml is not.
        An example can be found below::

            file_pairs = {'0722':{'model':{'wrfchem_v4.2':'/wrk/users/charkins/melodies-monet_data/wrfchem/run_CONUS_fv19_BEIS_1.0xISO_RACM_v4.2.2_racm_berk_vcp_noI_phot/0722/*'},
                          'obs':{'firexaq':'/wrk/d2/rschwantes/obs/firex-aq/R1/10s_merge/firexaq-mrg10-dc8_merge_20190722_R1.ict'}},
                '0905':{'model':{'wrfchem_v4.2':'/wrk/users/charkins/melodies-monet_data/wrfchem/run_CONUS_fv19_BEIS_1.0xISO_RACM_v4.2.2_racm_berk_vcp_noI_phot_soa/0905/*'},
                        'obs':{'firexaq':'/wrk/d2/rschwantes/obs/firex-aq/R1/10s_merge/firexaq-mrg10-dc8_merge_20190905_R1.ict'}}
                }

    file_pairs_yaml : str (optional)
        str containing path to a yaml file with file pairings.
        An example of the yaml file can be found in ``examples/yaml/supplementary_yaml/aircraft_looping_file_pairs.yaml``

    save_types : list (optional)
        List containing the types of data to save to netcdf. Can include any of 'paired', 'models', and 'obs'

    Returns
    -------
    None

    """
    from melodies_monet import driver

    if file_pairs_yaml:
        import yaml

        with open(file_pairs_yaml, "r") as stream:
            file_pairs = yaml.safe_load(stream)

    for file in file_pairs.keys():
        an = driver.analysis()
        an.control = control
        an.read_control()

        for model in an.control_dict["model"]:
            an.control_dict["model"][model]["files"] = file_pairs[file]["model"][model]
        for obs in an.control_dict["obs"]:
            an.control_dict["obs"][obs]["filename"] = file_pairs[file]["obs"][obs]

        an.control_dict["analysis"]["save"] = {}
        an.save = {}
        for t in save_types:
            an.control_dict["analysis"]["save"][t] = {
                "method": "netcdf",
                "prefix": file,
                "data": "all",
            }
            an.save[t] = {"method": "netcdf", "prefix": file, "data": "all"}

        an.open_models()
        an.open_obs()
        an.pair_data()
        an.save_analysis()


def convert_std_to_amb_ams(ds, convert_vars=[], temp_var=None, pres_var=None):

    # Convert variables from std to amb

    # Units of temp_var must be K
    # Units of pres_var must be Pa

    # So I just need to convert the obs from std to amb.
    # Losch = 2.69e25 # loschmidt's number
    # I checked the more detailed icart files
    # 273 K, 1 ATM (101325 Pa)
    std_ams = 101325.0 * N_A / (R * 273.0)
    # use pressure_obs now, which is in pa
    Airnum = ds[pres_var] * N_A / (R * ds[temp_var])

    # amb to std = Losch / Airnum
    convert_std_to_amb_ams = Airnum / std_ams

    for var in convert_vars:
        ds[var] = ds[var] * convert_std_to_amb_ams


def convert_std_to_amb_bc(ds, convert_vars=[], temp_var=None, pres_var=None):

    # Convert variables from std to amb

    # Units of temp_var must be K
    # Units of pres_var must be Pa

    # So I just need to convert the obs from std to amb.
    # Losch = 2.69e25 # loschmidt's number
    # 1013 mb, 273 K (101300 Pa)
    std_bc = 101300.0 * N_A / (R * 273.0)
    # use pressure_obs now, which is in pa
    Airnum = ds[pres_var] * N_A / (R * ds[temp_var])

    # amb to std = Losch / Airnum
    convert_std_to_amb_bc = Airnum / std_bc

    for var in convert_vars:
        ds[var] = ds[var] * convert_std_to_amb_bc


def calc_partialcolumn(ds: xr.Dataset, var: str = "NO2") -> xr.DataArray:
    """
    Calculates the partial column of a species from its concentration
    within a gridcell.

    Parameters
    ----------
    ds : xr.Dataset
        Model data containing `var`, 'pres_pa_mid', 'dz_m', and 'temperature_k'.
    var : str, optional
        Variable to calculate the partial column from, by default "NO2".

    Returns
    -------
    xr.DataArray
        DataArray containing the partial column of the species in molecules/cm2.
    """
    import pandas as pd

    ppbv2molmol = 1e-9
    m2_to_cm2 = 1e4
    fac_units = ppbv2molmol * N_A / m2_to_cm2
    partial_col = ds[var] * ds["pres_pa_mid"] * ds["dz_m"] * fac_units / (R * ds["temperature_k"])
    partial_col.attrs = {"units": "molecules/cm2", "long_name": f"{var} partial column"}

    # Update history for provenance
    history = ds.attrs.get("history", "")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["history"] = history + f" [{now}] Calculated {var} partial column."

    return partial_col


def calc_totalcolumn(ds: xr.Dataset, var: str = "NO2") -> xr.DataArray:
    """
    Calculates the total column of a species from its concentration.

    Parameters
    ----------
    ds : xr.Dataset
        Model data containing `var` and necessary vertical coordinates.
    var : str, optional
        Variable to calculate the total column from, by default "NO2".

    Returns
    -------
    xr.DataArray
        DataArray containing the total column of the species in molecules/cm2.
    """
    import pandas as pd

    data = calc_partialcolumn(ds, var)
    try:
        data = data.where(ds["pres_pa_mid"] <= ds["surfpres_pa"])
    except KeyError:
        pass
    total_col = data.sum(dim="z", keep_attrs=True)
    total_col.attrs = {"units": "molecules/cm2", "long_name": f"{var} total column"}

    # Update history for provenance
    history = ds.attrs.get("history", "")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["history"] = history + f" [{now}] Calculated {var} total column."

    return total_col


def calc_geolocaltime(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculates the geographic local time based on the longitude.

    Parameters
    ----------
    ds : xr.Dataset
        Model data containing 'longitude' and 'time'.

    Returns
    -------
    xr.DataArray
        DataArray containing the local time based on longitude.

    Examples
    --------
    >>> ds = xr.Dataset({"longitude": (("x",), [0, 15]), "time": pd.Timestamp("2023-01-01")})
    >>> local_time = calc_geolocaltime(ds)
    """
    import pandas as pd

    # 15 degrees = 1 hour = 3600 seconds -> 1 degree = 240 seconds
    offset_seconds = ds["longitude"] * 240

    # Use astype to preserve laziness
    delta = offset_seconds.astype("timedelta64[s]")

    localtime = ds["time"] + delta
    localtime.attrs["description"] = "Geographic local time, based on longitude"

    # Update history for provenance
    history = ds.attrs.get("history", "")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.attrs["history"] = history + f" [{now}] Calculated geolocaltime from longitude."

    return localtime
