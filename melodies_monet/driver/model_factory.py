# SPDX-License-Identifier: Apache-2.0
#
"""
The model factory class.
"""
import os
from .model import model


class ModelFactory:
    """
    Factory class for creating model objects.
    """

    def create_model(self, name, config, time_interval=None, load_files=True, control_dict=None):
        """
        Create and configure a model object.

        Parameters
        ----------
        name : str
            The name/label of the model.
        config : dict
            The configuration dictionary for the specific model.
        time_interval : list of pandas.Timestamp, optional
            Time interval to restrict the model data.
        load_files : bool, optional
            Whether to open the model files. Default is True.
        control_dict : dict, optional
            The full control dictionary, needed for some model operations.

        Returns
        -------
        model
            The configured model object.
        """
        m = model()
        m.model = config['mod_type']
        m.label = name

        if "is_global" in config:
            m.is_global = config['is_global']

        if 'radius_of_influence' in config:
            m.radius_of_influence = config['radius_of_influence']
        else:
            m.radius_of_influence = 1e6

        if 'mod_kwargs' in config:
            m.mod_kwargs = config['mod_kwargs']

        # create file string (note this can include hot strings)
        m.file_str = os.path.expandvars(config['files'])

        if 'files_vert' in config:
            m.file_vert_str = os.path.expandvars(config['files_vert'])
        if 'files_surf' in config:
            m.file_surf_str = os.path.expandvars(config['files_surf'])
        if 'files_pm25' in config:
            m.file_pm25_str = os.path.expandvars(config['files_pm25'])

        # create mapping
        m.mapping = config['mapping']

        # add variable dict
        if 'variables' in config:
            m.variable_dict = config['variables']
        if 'variable_summing' in config:
            m.variable_summing = config['variable_summing']
        if 'plot_kwargs' in config:
            m.plot_kwargs = config['plot_kwargs']

        # unstructured grid check
        if m.model in ['cesm_se']:
            if 'scrip_file' in config:
                m.scrip_file = config['scrip_file']
            else:
                raise ValueError('"Scrip_file" must be provided for unstructured grid output!')

        # maybe set projection
        self._set_projection(m, config.get("projection"))

        # open the model
        if load_files:
            # We pass control_dict because open_model_files uses it for some specific models like camx
            m.open_model_files(time_interval=time_interval, control_dict=control_dict)

        return m

    def _set_projection(self, m, proj_in):
        if proj_in == "None":
            print(
                f"NOTE: model.{m.label}.projection is {proj_in!r} (str), "
                "but we assume you want `None` (Python null sentinel). "
                "To avoid this warning, "
                "update your control file to remove the projection setting "
                "or set to `~` or `null` if you want null value in YAML."
            )
            proj_in = None

        if proj_in is not None:
            if isinstance(proj_in, str) and proj_in.startswith("model:"):
                m.proj = proj_in
            elif isinstance(proj_in, str) and proj_in.startswith("ccrs."):
                import cartopy.crs as ccrs
                m.proj = eval(proj_in)
            else:
                import cartopy.crs as ccrs
                if isinstance(proj_in, ccrs.Projection):
                    m.proj = proj_in
                else:
                    m.proj = ccrs.Projection(proj_in)
