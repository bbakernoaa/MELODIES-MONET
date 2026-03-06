import xarray as xr
import monet

class pair:
    """The pair class. Purely Xarray-based."""

    def __init__(self):
        self.obs = None
        self.model = None
        self.obj = None
        self.filename = None

    def pair_data(self, model_obj, obs_obj, **kwargs):
        """Standardized Xarray-to-Xarray pairing."""
        # This leverages the improved monet.pair which handles spatial/vertical/temporal
        return model_obj.monet.pair(obs_obj, **kwargs)
