# PR: [monetio] Migrate MM readers and utilities

## Summary
Migrates data-fetching and reading scripts from MELODIES-MONET into monetio.
- Implements `ICARTTReader` registered as both 'icartt' and 'aircraft'.
- Adds `GenericXarrayReader` for standard NetCDF files via `monetio.load`.
- Ports time-subsetting utilities (`subset_model_filelist`, `subset_OMPS_l2`, etc.) to `monetio.util`.

## Changes
- `monetio/readers/icartt.py`: Registered 'aircraft' name.
- `monetio/readers/generic_xarray.py`: Added generic NetCDF reader.
- `monetio/util.py`: Added subsetting functions.
