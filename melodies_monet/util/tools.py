# SPDX-License-Identifier: Apache-2.0
#
from __future__ import division

from builtins import range

import numpy as np
import pandas as pd
import xarray as xr

__author__ = "barry"


R = 8.31446261815324  # m3 * Pa / K / mol
N_A = 6.02214076e23


def search_listinlist(array1, array2):
    """
    Find intersections between two arrays and return their indices.

    Parameters
    ----------
    array1 : np.ndarray
        The first array.
    array2 : np.ndarray
        The second array.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing (indices_in_array1, indices_in_array2).
    """
    # find intersections

    s1 = set(array1.flatten())
    s2 = set(array2.flatten())

    inter = s1.intersection(s2)

    index1 = np.array([])
    index2 = np.array([])
    # find the indexes in array1
    for i in inter:
        index11 = np.where(array1 == i)
        index22 = np.where(array2 == i)
        index1 = np.concatenate([index1[:], index11[0]])
        index2 = np.concatenate([index2[:], index22[0]])

    return np.sort(np.int32(index1)), np.sort(np.int32(index2))


def list_contains(list1, list2):
    """Return True if any item in `list1` is also in `list2`."""
    for m in list1:
        for n in list2:
            if n == m:
                return True

    return False


def linregress(x, y):
    """
    Perform linear regression using Ordinary Least Squares.

    Parameters
    ----------
    x : array-like
        The independent variable.
    y : array-like
        The dependent variable.

    Returns
    -------
    tuple
        A tuple containing (slope, intercept, r_squared, std_err).
    """
    import statsmodels.api as sm

    xx = sm.add_constant(x)
    model = sm.OLS(y, xx)
    fit = model.fit()
    b, a = fit.params[0], fit.params[1]
    rsquared = fit.rsquared
    std_err = np.sqrt(fit.mse_resid)
    return a, b, rsquared, std_err


def findclosest(list, value):
    """Return (index, value) of the closest value in `list` to `value`."""
    a = min((abs(x - value), x, i) for i, x in enumerate(list))
    return a[2], a[1]


def _force_forder(x):
    """
    Converts arrays x to fortran order. Returns
    a tuple in the form (x, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


def kolmogorov_zurbenko_filter(df, col, window, iterations):
    """KZ filter implementation
    series is a pandas series
    window is the filter window m in the units of the data (m = 2q+1)
    iterations is the number of times the moving average is evaluated
    """
    df.index = df.time_local
    z = df.copy()
    for i in range(iterations):
        z.index = z.time_local
        z = z.groupby("siteid")[col].rolling(window, center=True, min_periods=1).mean(numeric_only=True).reset_index().dropna()
    df = df.reset_index(drop=True)
    return df.merge(z, on=["siteid", "time_local"])


def _wsdir2uv_numpy(ws, wdir):
    """Core NumPy implementation for wsdir2uv."""
    u = -ws * np.sin(np.deg2rad(wdir))
    v = -ws * np.cos(np.deg2rad(wdir))
    return u, v


def wsdir2uv(ws: xr.DataArray, wdir: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convert wind speed and direction to u and v components.

    Parameters
    ----------
    ws : xr.DataArray
        Wind speed.
    wdir : xr.DataArray
        Wind direction in degrees.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing (u_component, v_component).

    Examples
    --------
    >>> u, v = wsdir2uv(ws, wdir)
    """
    u, v = xr.apply_ufunc(
        _wsdir2uv_numpy,
        ws,
        wdir,
        input_core_dims=[[], []],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[ws.dtype, ws.dtype],
    )

    u.attrs["long_name"] = "u_wind_component"
    v.attrs["long_name"] = "v_wind_component"

    return u, v


def _get_relhum_numpy(temp, press, vap):
    """Core NumPy implementation for get_relhum."""
    temp_o = 273.16
    # Saturation vapor pressure (Pa) - Tetens formula
    es_vap = 611.0 * np.exp(17.67 * ((temp - temp_o) / (temp - 29.65)))
    # Saturation mixing ratio (kg/kg)
    ws_vap = 0.622 * (es_vap / press)
    relhum = 100.0 * (vap / ws_vap)
    return relhum


def get_relhum(temp: xr.DataArray, press: xr.DataArray, vap: xr.DataArray) -> xr.DataArray:
    """
    Calculate relative humidity from temperature, pressure, and water vapor mixing ratio.

    Parameters
    ----------
    temp : xr.DataArray
        Temperature in Kelvin.
    press : xr.DataArray
        Pressure in Pascals.
    vap : xr.DataArray
        Water vapor mixing ratio in kg/kg.

    Returns
    -------
    xr.DataArray
        Relative humidity in percent.

    Examples
    --------
    >>> relhum = get_relhum(temp, press, vap)
    """
    import pandas as pd

    relhum = xr.apply_ufunc(
        _get_relhum_numpy,
        temp,
        press,
        vap,
        dask="parallelized",
        output_dtypes=[temp.dtype],
    )

    relhum.attrs["units"] = "%"
    relhum.attrs["long_name"] = "relative_humidity"
    # Update history for provenance
    history = relhum.attrs.get("history", "")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    relhum.attrs["history"] = history + f" [{now}] Calculated relative humidity."

    return relhum


def long_to_wide(df):
    from pandas import merge

    w = df.pivot_table(values="obs", index=["time", "siteid"], columns="variable").reset_index()
    # cols = df.columns
    g = df.groupby("variable")
    for name, group in g:
        w[name + "_unit"] = group.units.unique()[0]
    # mergeon = hstack((index.values, df.variable.unique()))
    return merge(w, df, on=["siteid", "time"])


def calc_8hr_rolling_max(df, col=None, window=None):
    df.index = df.time_local
    df_rolling = (
        df.groupby("siteid")[col].rolling(window, center=True, win_type="boxcar").mean(numeric_only=True).reset_index().dropna()
    )
    df_rolling_max = df_rolling.groupby("siteid").resample("D", on="time_local").max(numeric_only=True).reset_index(drop=True)
    df = df.reset_index(drop=True)
    return df.merge(df_rolling_max, on=["siteid", "time_local"])


def calc_24hr_ave(df, col=None):
    df.index = df.time_local
    df_24hr_ave = df.groupby("siteid")[col].resample("D").mean(numeric_only=True).reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])


def calc_3hr_ave(df, col=None):
    df.index = df.time_local
    df_3hr_ave = df.groupby("siteid")[col].resample("3h").mean(numeric_only=True).reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_3hr_ave, on=["siteid", "time_local"])


def calc_annual_ave(df, col=None):
    df.index = df.time_local
    df_annual_ave = df.groupby("siteid")[col].resample("A").mean(numeric_only=True).reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_annual_ave, on=["siteid", "time_local"])


def get_giorgi_region_bounds(index=None, acronym=None):
    import pandas as pd

    i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    acro = [
        "NAU",
        "SAU",
        "AMZ",
        "SSA",
        "CAM",
        "WNA",
        "CNA",
        "ENA",
        "ALA",
        "GRL",
        "MED",
        "NEU",
        "WAF",
        "EAF",
        "SAF",
        "SAH",
        "SEA",
        "EAS",
        "SAS",
        "CAS",
        "TIB",
        "NAS",
    ]
    lonmax = [
        155,
        155,
        -34,
        -40,
        -83,
        -103,
        -85,
        -60,
        -103,
        -10,
        40,
        40,
        22,
        52,
        52,
        65,
        155,
        145,
        100,
        75,
        100,
        180,
    ]
    lonmin = [
        110,
        110,
        -82,
        -76,
        -116,
        -130,
        -103,
        -85,
        -170,
        -103,
        -10,
        -10,
        -20,
        22,
        -10,
        -20,
        95,
        100,
        65,
        40,
        75,
        40,
    ]
    latmax = [
        -11,
        -28,
        12,
        -20,
        30,
        60,
        50,
        50,
        72,
        85,
        48,
        75,
        18,
        18,
        -12,
        30,
        20,
        50,
        30,
        50,
        50,
        70,
    ]
    latmin = [
        -28,
        -45,
        -20,
        -56,
        10,
        30,
        30,
        25,
        60,
        50,
        30,
        48,
        -12,
        -12,
        -35,
        18,
        -11,
        20,
        5,
        30,
        30,
        50,
    ]
    df = pd.DataFrame(
        {
            "latmin": latmin,
            "lonmin": lonmin,
            "latmax": latmax,
            "lonmax": lonmax,
            "acronym": acro,
        },
        index=i,
    )
    try:
        if index is None and acronym is None:
            print("either index or acronym needs to be supplied")
            print("look here https://web.northeastern.edu/sds/web/demsos/images_002/subregions.jpg")
            raise ValueError
        elif index is not None:
            return df.loc[df.index == index].values.flatten()
        else:
            return df.loc[df.acronym == acronym.upper()].values.flatten()
    except ValueError:
        exit


def get_giorgi_region_df(df):
    df.loc[:, "GIORGI_INDEX"] = None
    df.loc[:, "GIORGI_ACRO"] = None
    for i in range(22):
        latmin, lonmin, latmax, lonmax, acro = get_giorgi_region_bounds(index=int(i + 1))
        con = (df.longitude <= lonmax) & (df.longitude >= lonmin) & (df.latitude <= latmax) & (df.latitude >= latmin)
        df.loc[con, "GIORGI_INDEX"] = i + 1
        df.loc[con, "GIORGI_ACRO"] = acro
    return df


def get_epa_region_bounds(index=None, acronym=None):
    import pandas as pd

    i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    acro = [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
        "AK",
        "PR",
        "VI",
    ]
    lonmax = [
        -66.8628,
        -73.8885,
        -74.8526,
        -75.4129,
        -80.5188,
        -88.7421,
        -89.1005,
        -96.438,
        -109.0475,
        -111.0471,
        -129.99,
        -65.177765,
        -64.26384,
    ]
    lonmin = [
        -73.7272,
        -79.7624,
        -83.6753,
        -91.6589,
        -97.2304,
        -109.0489,
        -104.0543,
        -116.0458,
        -124.6509,
        -124.7305,
        -169.9146,
        -67.289886,
        -64.861221,
    ]
    latmax = [
        47.455,
        45.0153,
        42.5167,
        39.1439,
        49.3877,
        37.0015,
        43.5008,
        48.9991,
        42.0126,
        49.0027,
        71.5232,
        18.520551,
        18.751244,
    ]
    latmin = [
        40.9509,
        38.8472,
        36.5427,
        24.3959,
        36.9894,
        25.8419,
        35.9958,
        36.9949,
        31.3325,
        41.9871,
        52.5964,
        17.904834,
        18.302014,
    ]
    df = pd.DataFrame(
        {
            "latmin": latmin,
            "lonmin": lonmin,
            "latmax": latmax,
            "lonmax": lonmax,
            "acronym": acro,
        },
        index=i,
    )
    try:
        if index is None and acronym is None:
            print("either index or acronym needs to be supplied")
            print(
                "Look here for more information: https://www.epa.gov/enviro/epa-regional-kml-download",
                "https://gist.github.com/jakebathman/719e8416191ba14bb6e700fc2d5fccc5",
            )
            raise ValueError
        elif index is not None:
            return df.loc[df.index == index].values.flatten()
        else:
            return df.loc[df.acronym == acronym.upper()].values.flatten()
    except ValueError:
        exit


def get_epa_region_df(df):
    df.loc[:, "EPA_INDEX"] = None
    df.loc[:, "EPA_ACRO"] = None
    for i in range(13):
        latmin, lonmin, latmax, lonmax, acro = get_epa_region_bounds(index=int(i + 1))
        con = (df.longitude <= lonmax) & (df.longitude >= lonmin) & (df.latitude <= latmax) & (df.latitude >= latmin)
        df.loc[con, "EPA_INDEX"] = i + 1
        df.loc[con, "EPA_ACRO"] = acro
    return df


def resample_stratify(da, levels, vertical, axis=1, interpolation="linear", extrapolation="nan"):
    import stratify

    result = stratify.interpolate(
        levels,
        vertical.chunk().data,
        da.chunk().data,
        axis=axis,
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    dims = da.dims
    out = xr.DataArray(result, dims=dims)
    for i in dims:
        if i != "z":
            out[i] = da[i]
    out.attrs = da.attrs.copy()
    if len(da.coords) > 0:
        for i in da.coords:
            if i != "z":
                out.coords[i] = da.coords[i]
    return out


def vert_interp(ds_model: xr.Dataset, df_obs: pd.DataFrame, var_name_list: list) -> pd.DataFrame:
    """
    Perform vertical interpolation for pairing.

    Parameters
    ----------
    ds_model : xr.Dataset
        Model data.
    df_obs : pd.DataFrame
        Observation data.
    var_name_list : list
        List of variables to interpolate.

    Returns
    -------
    pd.DataFrame
        Paired data.
    """
    from pandas import merge_asof

    ds_model["pressure_model_nan"] = ds_model["pressure_model"].copy()
    var_name_list.append("pressure_model_nan")

    # Optimization: Extract pressure values once
    p_obs_vals = sorted(ds_model.pressure_obs.squeeze().values, reverse=True)

    var_out_list = []
    for var_name in var_name_list:
        if var_name == "pressure_model":
            extrap = "linear"
        elif var_name == "pressure_model_nan":
            extrap = "nan"
        else:
            extrap = "nearest"

        out = resample_stratify(
            ds_model[var_name],
            p_obs_vals,
            ds_model["pressure_model"],
            axis=1,
            interpolation="linear",
            extrapolation=extrap,
        )
        out.name = var_name
        var_out_list.append(out)

    # Conversion to dataframe is unavoidable for merge_asof,
    # but we minimize the data being converted.
    df_model = xr.merge(var_out_list).to_dataframe().reset_index()

    # Clean up and rename
    df_model.drop(
        labels=["x", "y", "z", "pressure_obs", "pressure_model_nan", "time_obs"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    df_model.rename(columns={"pressure_model": "pressure_obs"}, inplace=True)

    final_df_model = merge_asof(
        df_obs,
        df_model,
        by=["latitude", "longitude", "pressure_obs"],
        on="time",
        direction="nearest",
    )

    return final_df_model


def mobile_and_ground_pair(ds_model: xr.Dataset, df_obs: pd.DataFrame, var_name_list: list) -> pd.DataFrame:
    """
    Pair mobile or ground-based observations with model data.

    Parameters
    ----------
    ds_model : xr.Dataset
        Model data.
    df_obs : pd.DataFrame
        Observation data.
    var_name_list : list
        List of variables to pair.

    Returns
    -------
    pd.DataFrame
        Paired data.
    """
    from pandas import merge_asof

    # Extract just the surface level data
    if "z" in ds_model.dims:
        ds_surf = ds_model[var_name_list].isel(z=0)
    else:
        print("WARNING: No z dimension in model, assuming all data at surface.")
        ds_surf = ds_model[var_name_list]

    df_model = ds_surf.to_dataframe().reset_index()

    # Drop non-essential columns before merge to save memory
    df_model.drop(labels=["x", "y", "time_obs"], axis=1, inplace=True, errors="ignore")

    final_df_model = merge_asof(
        df_obs,
        df_model,
        by=["latitude", "longitude"],
        on="time",
        direction="nearest",
        suffixes=("", "_new"),
    )

    return final_df_model


def find_obs_time_bounds(files=[], time_var=None):
    """Function to read a series of ict files and print a list of min and max times for each.

    Parameters
    ----------
    files : str or iterable
        str or list of str containing filenames that should be read.

    time_var : str
        Optional, variable name that should be assumed to be time when reading aircraft csv files.

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
    for file in files:
        _, extension = os.path.splitext(files[0])
        try:
            if extension in {".nc", ".ncf", ".netcdf", ".nc4"}:
                obs = xr.open_dataset(file)
            elif extension in [".ict", ".icartt"]:
                obs = mio.icartt.add_data(file)
            elif extension in [".csv"]:
                import monetio as mio

                obs = mio.load("aircraft_csv", files=file, time_var=time_var)
            else:
                raise ValueError(f"extension {extension!r} currently unsupported")
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
