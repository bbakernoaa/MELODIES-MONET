# SPDX-License-Identifier: Apache-2.0
#
"""
file: regrid_util.py
"""

import os

import xarray as xr


def setup_regridder(config, config_group="obs", target_grid=None):
    """
    Setup regridder for observations or model.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    config_group : str, optional
        The configuration group to use ('obs' or 'model'). Default is 'obs'.
    target_grid : xr.Dataset, optional
        Target grid dataset. If not provided, it is read from the configuration.

    Returns
    -------
    regridder_dict : dict of xesmf.Regridder
        Dictionary of regridder instances.
    """
    try:
        import xesmf as xe
    except ImportError:
        print("regrid_util: xesmf module not found")
        raise

    print("setup_regridder.target_grid")
    print(target_grid)

    if target_grid is not None:
        ds_target = target_grid
    else:
        target_file = os.path.expandvars(config["analysis"]["target_grid"])
        ds_target = xr.open_dataset(target_file)

    regridder_dict = dict()

    for name in config[config_group]:
        base_file = os.path.expandvars(config[config_group][name]["regrid"]["base_grid"])
        ds_base = xr.open_dataset(base_file)
        method = config[config_group][name]["regrid"]["method"]
        regridder = xe.Regridder(ds_base, ds_target, method)
        regridder_dict[name] = regridder

    return regridder_dict


def filename_regrid(filename, regridder):
    """
    Construct modified filename for regridded dataset.

    Parameters
    ----------
    filename : str
        Filename of dataset.
    regridder : xesmf.Regridder
        Regridder instance.

    Returns
    -------
    str
        Filename of regridded dataset.
    """
    filename_regrid = filename.replace(".nc", "_regrid.nc")

    return filename_regrid
