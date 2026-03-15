Downloading Observations
========================

As described below, some observations can be directly used in the MELODIES MONET tool as is 
and some need a preprocessing step to convert them into a consistent data format.

The MELODIES-MONET orchestrator handles data loading and preprocessing natively via `monetio`.
Users can specify the data sources and time periods in the control YAML file, and the tool will
automatically fetch, load, and standardize the datasets (including AirNow, AERONET, AQS, ISH,
and OpenAQ) for the analysis.

Aircraft, Sonde, Mobile, and Ground Campaign Data
-------------------------------------------------

Aircraft, sonde, mobile, and ground campaign data can be used directly in the tool as long 
as the data format is NetCDF, `ICARTT <https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm>`_, or CSV.

Satellite
---------

For satellite data, the tool supports common Air Quality and Atmospheric Composition products
via `monetio` readers. Metadata and swath-to-grid interpolation are handled automatically
during the pairing process.
