# SPDX-License-Identifier: Apache-2.0
#
"""
Factory for creating and configuring model objects.
"""
import os
import cartopy.crs as ccrs
from .model import model

class ModelFactory:
    def __init__(self):
        pass

    def create_model(self, model_name, model_config, control_dict=None):
        """
        Create and configure a model object.

        Parameters
        ----------
        model_name : str
            The label/name of the model.
        model_config : dict
            The configuration dictionary for this specific model.
        control_dict : dict, optional
            The full control dictionary, needed for some model settings (like passing to open_model_files).

        Returns
        -------
        model
            The configured model object.
        """
        m = model()
        m.label = model_name

        # model type (ie cmaq, rapchem, gsdchem etc)
        m.model = model_config['mod_type']

        # set the model label in the dictionary and model class instance
        if "is_global" in model_config:
            m.is_global = model_config['is_global']

        m.radius_of_influence = model_config.get('radius_of_influence', 1e6)
        m.mod_kwargs = model_config.get('mod_kwargs', {})

        # create file string (note this can include hot strings)
        m.file_str = os.path.expandvars(model_config['files'])

        if 'files_vert' in model_config:
            m.file_vert_str = os.path.expandvars(model_config['files_vert'])
        if 'files_surf' in model_config:
            m.file_surf_str = os.path.expandvars(model_config['files_surf'])
        if 'files_pm25' in model_config:
            m.file_pm25_str = os.path.expandvars(model_config['files_pm25'])

        # create mapping
        m.mapping = model_config['mapping']

        # add variable dict
        if 'variables' in model_config:
            m.variable_dict = model_config['variables']
        if 'variable_summing' in model_config:
            m.variable_summing = model_config['variable_summing']
        if 'plot_kwargs' in model_config:
            m.plot_kwargs = model_config['plot_kwargs']

        # unstructured grid check
        if m.model in ['cesm_se']:
            if 'scrip_file' in model_config:
                m.scrip_file = model_config['scrip_file']
            else:
                raise ValueError('"Scrip_file" must be provided for unstructured grid output!')

        # maybe set projection
        self._set_projection(m, model_config)

        return m

    def _set_projection(self, m, model_config):
        proj_in = model_config.get("projection")
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
                m.proj = eval(proj_in)
            else:
                if isinstance(proj_in, ccrs.Projection):
                    m.proj = proj_in
                else:
                    m.proj = ccrs.Projection(proj_in)
