Downloading Observations
========================

As described below, some observations can be directly used in the MELODIES MONET tool as is 
and some need a preprocessing step to convert them into a consistent data format.

Surface
-------

Surface datasets commonly used for air quality and atmospheric composition applications are all in different 
formats and occasionally some HPC platforms including the NOAA Hera machine have download restrictions 
that prevent us from using the automatic download features available in MONET. For these reasons,
it is often useful to download and preprocess the surface observational datasets and save the output to an
intermediate NetCDF file. These files can then be reused, avoiding the need to re-download
observational data over and over again for the same analysis period.

The :mod:`monetio` package provides a Command Line Interface (CLI) that can be used to download and create
standardized datasets for: AirNow, AERONET, AQS, and OpenAQ.

The Command Line Interface allows users to very easily download datasets with a single command.
Generally, users only need to select which subcommand to use (i.e., which observational data set you want to download) 
and then specify the date range like that below to download US EPA AQS observations::

    $ monetio aqs -d 2023-08-01:2023-08-31 -p OZONE -p PM2.5 -o epa_data.nc

The other datasets can be downloaded in a similar way::

    $ monetio aeronet -d 2023-08-01:2023-08-31 -o aeronet_data.nc
    $ monetio airnow -d 2023-08-01:2023-08-31 -o airnow_data.nc --wide-fmt
    $ monetio openaq -d 2023-08-01:2023-08-31 -o openaq_data.nc

For more information and options, please refer to the `MONETIO CLI documentation <https://monetio.readthedocs.io/en/stable/cli.html>`_.

.. note::
   For users using MELODIES MONET on the NOAA Hera machine (or other machines 
   with download restrictions), you will need to use the MONETIO Command Line Interface on a
   machine without download restrictions and manually copy the netCDF files produced 
   onto the NOAA Hera machine.

Aircraft, Sonde, Mobile, and Ground Campaign Data
-------------------------------------------------

Aircraft, sonde, mobile, and ground campaign data can be used directly in the tool as long 
as the data format is NetCDF, `ICARTT <https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm>`_, or CSV. Users download their own observational data. 
No pre-processing is required for these datasets.

Satellite
---------

For satellite data, users download their own observational data. No pre-processing is required 
for these datasets.
