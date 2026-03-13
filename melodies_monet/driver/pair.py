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
        """Reformat the paired dataset. Redirection to monet.util.combinetool.fix_paired_xarray."""
        import monet as m

        return m.util.combinetool.fix_paired_xarray(dset)
