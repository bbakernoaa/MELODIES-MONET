Downloading Observations
========================

As described below, some observations can be directly used in the MELODIES MONET tool as is 
and some need a preprocessing step to convert them into a consistent data format.

Surface
-------

Surface datasets commonly used for air quality and atmospheric composition applications are all in different 
formats and occasionally some HPC platforms including the NOAA Hera machine have download restrictions 
that prevent us from using the automatic download features available in MONET. So for now, 
MELODIES MONET has separate scripts to preprocess the surface observational datasets and save the output to an 
intermediate NetCDF file. These preprocess scripts are also useful so that users do not have to re-download 
observational data over and over again for the same analysis period. We will work on automating this process further 
in the future.

MELODIES MONET leverages :mod:`monetio` to directly download and load many surface observational datasets. Users no longer need to use separate CLI commands for downloading; instead, specify the dataset source (e.g., 'airnow') in the YAML configuration file.

For more details on available sources and configuration, see :doc:`/users_guide/supported_datasets`.

.. note::
   On platforms with strict download restrictions (like NOAA Hera), you may need to download data on a machine with internet access and transfer the files manually.

Aircraft, Sonde, Mobile, and Ground Campaign Data
-------------------------------------------------

Aircraft, sonde, mobile, and ground campaign data can be used directly in the tool as long 
as the data format is NetCDF, `ICARTT <https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm>`_, or CSV. Users download their own observational data. 
No pre-processing is required for these datasets.

Satellite
---------

For satellite data, users download their own observational data. No pre-processing is required 
for these datasets.