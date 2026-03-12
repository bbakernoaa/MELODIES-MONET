# SPDX-License-Identifier: Apache-2.0
#
import xarray as xr
import pandas as pd


class pair:
    """The pair class.

    The pair class pairs model data
    directly with observational data along time and space.
    """

    def __init__(self):
        """Initialize a :class:`pair` object."""
        self.type = "pt_sfc"
        self.radius_of_influence = 1e6
        self.obs = None
        self.model = None
        self.model_vars = None
        self.obs_vars = None
        self.filename = None

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    type={self.type!r},\n"
            f"    radius_of_influence={self.radius_of_influence!r},\n"
            f"    obs={self.obs!r},\n"
            f"    model={self.model!r},\n"
            f"    model_vars={self.model_vars!r},\n"
            f"    obs_vars={self.obs_vars!r},\n"
            f"    filename={self.filename!r},\n"
            ")"
        )

    def fix_paired_xarray(self, dset):
        """Reformat the paired dataset using xarray operations to maintain laziness.

        Parameters
        ----------
        dset : xarray.Dataset

        Returns
        -------
        xarray.Dataset
            Reformatted paired dataset.
        """
        if "siteid" not in dset.coords and "siteid" not in dset.data_vars:
             return dset

        # Ensure we have a site dimension 'x'
        if "x" not in dset.dims:
            dset = dset.rename({"siteid": "x"})

        site_columns = [
            "latitude",
            "longitude",
            "site",
            "msa_code",
            "cmsa_name",
            "epa_region",
            "state_name",
            "msa_name",
            "utcoffset",
        ]

        # Separate site-invariant and time-variant data
        # Site data is constant over time for each site
        site_vars = [v for v in site_columns if v in dset.data_vars]

        # Create a site-only dataset by taking the first time step for site variables
        # and dropping the time dimension
        ds_site = dset[site_vars].isel(time=0, drop=True)

        # Create a time-variant dataset by dropping the site variables
        ds_time = dset.drop_vars(site_vars)

        # Merge them back. Xarray will handle the broadcasting correctly.
        out = xr.merge([ds_time, ds_site])

        # Ensure siteid is preserved as a coordinate or variable
        if "siteid" not in out.coords and "siteid" not in out.data_vars:
             if "x" in out.dims:
                  # If we renamed siteid to x, we might want to keep siteid as a variable
                  pass

        return out
