# monetio PR: Migrate MM Readers and Standardize Xarray Output

## Description
This PR migrates custom data-fetching and reading scripts from `melodies_monet` into `monetio/readers` using the new class inheritance structure. It ensures all readers return standardized Xarray objects, facilitating seamless integration with the MELODIES-MONET DAG.

## Changes
- Integrated `read_aircraft_obs_csv` as `AircraftCSVReader` in `monetio.readers.aircraft_csv`.
- Migrated time-subsetting utilities (`subset_model_filelist`, `subset_OMPS_l2`, etc.) to `monetio.util`.
- Standardized all readers to return UGRID-compliant Xarray Datasets.
