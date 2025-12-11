# SPDX-License-Identifier: Apache-2.0
#
"""
The analysis class.
"""
import os
import yaml
import pandas as pd
import numpy as np
import xarray as xr
import datetime

from . import _pairing_strategies as ps
from .model import model
from .model_factory import ModelFactory
from .observation import observation
from .observation_factory import ObservationFactory
from .obs_gridder import ObsGridder
from .pair import pair


class analysis:
    """The analysis class.

    The analysis class is the highest
    level class and stores all information about the analysis. It reads
    and stores information from the input yaml file and defines
    overarching analysis information like the start and end time, which
    models and observations to pair, etc.
    """

    def __init__(self):
        """Initialize an :class:`analysis` object."""
        self.control = 'control.yaml'
        self.control_dict = None
        self.models = {}
        """dict : Models, set by :meth:`open_models`."""
        self.obs = {}
        """dict : Observations, set by :meth:`open_obs`."""
        self.paired = {}
        """dict : Paired data, set by :meth:`pair_data`."""
        self.start_time = None
        self.end_time = None
        self.time_intervals = None
        self.download_maps = True  # Default to True
        self.output_dir = None
        self.output_dir_save = None
        self.output_dir_read = None
        self.debug = False
        self.save = None
        self.read = None
        self.regrid = False  # Default to False
        self.target_grid = None
        self.obs_regridders = None
        self.model_regridders = None
        self.obs_gridder = None
        self.add_logo = True
        """bool, default=True : Add the MELODIES MONET logo to the plots."""
        self.pairing_kwargs = {}


    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    control={self.control!r},\n"
            f"    control_dict={repr(self.control_dict) if self.control_dict is None else '...'},\n"
            f"    models={self.models!r},\n"
            f"    obs={self.obs!r},\n"
            f"    paired={self.paired!r},\n"
            f"    start_time={self.start_time!r},\n"
            f"    end_time={self.end_time!r},\n"
            f"    time_intervals={self.time_intervals!r},\n"
            f"    download_maps={self.download_maps!r},\n"
            f"    output_dir={self.output_dir!r},\n"
            f"    output_dir_save={self.output_dir_save!r},\n"
            f"    output_dir_read={self.output_dir_read!r},\n"
            f"    debug={self.debug!r},\n"
            f"    save={self.save!r},\n"
            f"    read={self.read!r},\n"
            f"    regrid={self.regrid!r},\n"
            ")"
        )

    def read_control(self, control=None):
        """Read the input yaml file,
        updating various :class:`analysis` instance attributes.

        Parameters
        ----------
        control : str
            Input yaml file path.
            If provided, :attr:`control` will be set to this value.

        Returns
        -------
        type
            Reads the contents of the yaml control file into a dictionary associated with the analysis class.
        """
        import yaml

        if control is not None:
            self.control = control

        with open(self.control, 'r') as stream:
            self.control_dict = yaml.safe_load(stream)

        # set analysis time
        if 'start_time' in self.control_dict['analysis'].keys():
            self.start_time = pd.Timestamp(self.control_dict['analysis']['start_time'])
        if 'end_time' in self.control_dict['analysis'].keys():
            self.end_time = pd.Timestamp(self.control_dict['analysis']['end_time'])
        if 'output_dir' in self.control_dict['analysis'].keys():
            self.output_dir = os.path.expandvars(
                    self.control_dict['analysis']['output_dir'])
        else:
            raise Exception('output_dir was not specified and is required. Please set analysis.output_dir in the control file.')
        if 'output_dir_save' in self.control_dict['analysis'].keys():
            self.output_dir_save = os.path.expandvars(
                self.control_dict['analysis']['output_dir_save'])
        else:
            self.output_dir_save=self.output_dir
        if 'output_dir_read' in self.control_dict['analysis'].keys():
            if self.control_dict['analysis']['output_dir_read'] is not None:
                self.output_dir_read = os.path.expandvars(
                    self.control_dict['analysis']['output_dir_read'])
        else:
            self.output_dir_read=self.output_dir

        self.debug = self.control_dict['analysis']['debug']
        if 'save' in self.control_dict['analysis'].keys():
            self.save = self.control_dict['analysis']['save']
        if 'read' in self.control_dict['analysis'].keys():
            self.read = self.control_dict['analysis']['read']
        if 'add_logo' in self.control_dict['analysis'].keys():
            self.add_logo = self.control_dict['analysis']['add_logo']

        if 'regrid' in self.control_dict['analysis'].keys():
            self.regrid = self.control_dict['analysis']['regrid']
        if 'target_grid' in self.control_dict['analysis'].keys():
            self.target_grid = self.control_dict['analysis']['target_grid']

        # generate time intervals for time chunking
        if 'time_interval' in self.control_dict['analysis'].keys():
            time_stamps = pd.date_range(
                start=self.start_time, end=self.end_time,
                freq=self.control_dict['analysis']['time_interval'])
            # if (end_time - start_time) is not an integer multiple
            #   of freq, append end_time to time_stamps
            if time_stamps[-1] < pd.Timestamp(self.end_time):
                time_stamps = time_stamps.append(
                    pd.DatetimeIndex([self.end_time]))
            self.time_intervals \
                = [[time_stamps[n], time_stamps[n+1]]
                    for n in range(len(time_stamps)-1)]

        # specific arguments for pairing options
        if 'pairing_kwargs' in self.control_dict['analysis'].keys():
            self.pairing_kwargs = self.control_dict['analysis']['pairing_kwargs']

        # Enable Dask progress bars? (default: false)
        enable_dask_progress_bars = self.control_dict["analysis"].get(
            "enable_dask_progress_bars", False)
        if enable_dask_progress_bars:
            from dask.diagnostics import ProgressBar

            ProgressBar().register()
        else:
            from dask.callbacks import Callback

            Callback.active = set()

    def save_analysis(self):
        """Save all analysis attributes listed in analysis section of input yaml file.

        Returns
        -------
        None
        """
        if self.save is not None:
            # Loop over each possible attr type (models, obs and paired)
            for attr in self.save:
                if self.save[attr]['method']=='pkl':
                    from ..util.write_util import write_pkl
                    write_pkl(obj=getattr(self,attr), output_name=os.path.join(self.output_dir_save,self.save[attr]['output_name']))

                elif self.save[attr]['method']=='netcdf':
                    from ..util.write_util import write_analysis_ncf
                    # save either all groups or selected groups
                    if self.save[attr]['data']=='all':
                        if 'prefix' in self.save[attr]:
                            write_analysis_ncf(obj=getattr(self,attr), output_dir=self.output_dir_save,
                                               fn_prefix=self.save[attr]['prefix'])
                        else:
                            write_analysis_ncf(obj=getattr(self,attr), output_dir=self.output_dir_save)
                    else:
                        if 'prefix' in self.save[attr]:
                            write_analysis_ncf(obj=getattr(self,attr), output_dir=self.output_dir_save,
                                               fn_prefix=self.save[attr]['prefix'], keep_groups=self.save[attr]['data'])
                        else:
                            write_analysis_ncf(obj=getattr(self,attr), output_dir=self.output_dir_save,
                                               keep_groups=self.save[attr]['data'])

    def read_analysis(self):
        """Read all previously saved analysis attributes listed in analysis section of input yaml file.

        Returns
        -------
        None
        """
        if self.read is not None:
            # Loop over each possible attr type (models, obs and paired)
            from ..util.read_util import read_saved_data
            for attr in self.read:
                if self.read[attr]['method']=='pkl':
                    read_saved_data(analysis=self,filenames=self.read[attr]['filenames'], method='pkl', attr=attr)
                elif self.read[attr]['method']=='netcdf':
                    read_saved_data(analysis=self,filenames=self.read[attr]['filenames'], method='netcdf', attr=attr)
                if attr == 'paired':
                    # initialize model/obs attributes, since needed for plotting and stats
                    if not self.models:
                        self.open_models(load_files=False)
                    if not self.obs:
                        self.open_obs(load_files=False)

    def setup_regridders(self):
        """Create an obs xesmf.Regridder from base and target grids specified in the control_dict

        Returns
        -------
        None
        """
        from ..util import regrid_util
        if self.regrid:
            if self.target_grid == 'obs_grid':
                self.model_regridders = regrid_util.setup_regridder(self.control_dict, config_group='model', target_grid=self.da_obs_grid)
            else:
                self.obs_regridders = regrid_util.setup_regridder(self.control_dict, config_group='obs')
                self.model_regridders = regrid_util.setup_regridder(self.control_dict, config_group='model')

    def open_models(self, time_interval=None,load_files=True):
        """Open all models listed in the input yaml file and create a :class:`model`
        object for each of them, populating the :attr:`models` dict.

        Parameters
        ----------
        time_interval (optional, default None) : [pandas.Timestamp, pandas.Timestamp]
            If not None, restrict models to datetime range spanned by time interval [start, end].
        load_files (optional, default True): boolean
            If False, only populate :attr: dict with yaml file parameters and do not open model files.
        Returns
        -------
        None
        """
        if 'model' in self.control_dict:
            factory = ModelFactory()
            # open each model
            for mod in self.control_dict['model']:
                model_config = self.control_dict['model'][mod]
                m = factory.create_model(mod, model_config, self.control_dict)

                # open the model
                if load_files:
                    m.open_model_files(time_interval=time_interval, control_dict=self.control_dict)
                self.models[m.label] = m

    def open_obs(self, time_interval=None, load_files=True):
        """Open all observations listed in the input yaml file and create an
        :class:`observation` instance for each of them,
        populating the :attr:`obs` dict.

        Parameters
        ----------
        time_interval (optional, default None) : [pandas.Timestamp, pandas.Timestamp]
            If not None, restrict obs to datetime range spanned by time interval [start, end].
        load_files (optional, default True): boolean
            If False, only populate :attr: dict with yaml file parameters and do not open obs files.

        Returns
        -------
        None
        """
        if 'obs' in self.control_dict:
            factory = ObservationFactory()
            for obs in self.control_dict['obs']:
                obs_config = self.control_dict['obs'][obs]
                o = factory.create_observation(obs, obs_config, self.control_dict)

                if load_files:
                    if o.obs_type in ['sat_swath_sfc', 'sat_swath_clm', 'sat_grid_sfc',\
                                        'sat_grid_clm', 'sat_swath_prof']:
                        o.open_sat_obs(time_interval=time_interval, control_dict=self.control_dict)
                    else:
                        o.open_obs(time_interval=time_interval, control_dict=self.control_dict)
                self.obs[o.label] = o

    def setup_obs_grid(self):
        """
        Setup a uniform observation grid.
        """
        self.obs_gridder = ObsGridder(self.control_dict)
        self.obs_gridder.setup_obs_grid()
        self.da_obs_grid = self.obs_gridder.da_obs_grid

    def update_obs_gridded_data(self):
        """
        Update observation grid cell values and counts,
        for all observation datasets and parameters.
        """
        if self.obs_gridder:
             self.obs_gridder.update_obs_gridded_data(self.obs)

    def normalize_obs_gridded_data(self):
        """
        Normalize observation grid cell values where counts is not zero.
        Create data arrays for the obs_gridded_dataset dictionary.
        """
        if self.obs_gridder:
             self.obs_gridded_dataset = self.obs_gridder.normalize_obs_gridded_data()

    def pair_data(self, time_interval=None):
        """Pair all observations and models in the analysis class
        (i.e., those listed in the input yaml file) together,
        populating the :attr:`paired` dict.

        Parameters
        ----------
        time_interval (optional, default None) : [pandas.Timestamp, pandas.Timestamp]
            If not None, restrict pairing to datetime range spanned by time interval [start, end].


        Returns
        -------
        None
        """
        print('1, in pair data')
        for model_label in self.models:
            mod = self.models[model_label]
            for obs_to_pair in mod.mapping.keys():
                keys = list(mod.mapping[obs_to_pair].keys())
                obs_vars = [mod.mapping[obs_to_pair][key] for key in keys]

                mod_vars = list(mod.variable_dict.keys()) if mod.variable_dict is not None else []

                if mod.obj.attrs.get("mio_scrip_file", False):
                    lonlat_list = ['lon', 'lat', 'longitude', 'latitude', 'Longitude', 'Latitude']
                    for ll in lonlat_list:
                        if ll in mod.obj.data_vars:
                            keys.append(ll)

                model_obj = mod.obj[keys + mod_vars]
                obs = self.obs[obs_to_pair]

                obs_type = obs.obs_type.lower()
                if obs_type == 'pt_sfc':
                    p = ps._pair_pt_sfc(model_obj, mod, obs, keys, obs_vars, self.debug)
                elif obs_type == 'aircraft':
                    p = ps._pair_aircraft(model_obj, mod, obs, keys, obs_vars, mod_vars)
                elif obs_type == 'sonde':
                    p = ps._pair_sonde(
                        model_obj, mod, obs, keys, obs_vars, mod_vars, self.control_dict
                    )
                elif obs_type in ('mobile', 'ground'):
                    p = ps._pair_mobile_or_ground(model_obj, mod, obs, keys, obs_vars, mod_vars)
                elif obs_type == 'sat_swath_clm':
                    p = ps._pair_sat_swath_clm(
                        model_obj,
                        mod,
                        obs,
                        keys,
                        obs_vars,
                        self.start_time,
                        self.end_time,
                        self.pairing_kwargs,
                    )
                elif obs_type == 'sat_grid_clm':
                    p = ps._pair_sat_grid_clm(
                        model_obj,
                        mod,
                        obs,
                        keys,
                        obs_vars,
                        self.start_time,
                        self.end_time,
                        self.pairing_kwargs,
                    )
                else:
                    # TODO: add other network types / data types where (ie flight, satellite etc)
                    raise ValueError(f"Unknown obs type for pairing: {obs.obs_type!r}")

                label = f"{p.obs}_{p.model}"
                self.paired[label] = p

    def concat_pairs(self):
        """Read and concatenate all observation and model time interval pair data,
        populating the :attr:`paired` dict.

        Returns
        -------
        None
        """
        pass

    def plotting(self):
        from . import plotting
        plotting.plotting(self)

    def stats(self):
        from . import stats
        stats.stats(self)
