# MELODIES-MONET Driver Migration Map

This document provides a comprehensive static analysis and mapping of the MELODIES-MONET driver code. The goal is to facilitate the transition to a more unified library architecture by categorizing existing functions and identifying areas for abstraction.

## 1. Core Dependency Matrix

The following table categorizes every function in the `melodies_monet/driver/` directory based on its primary logic and its reliance on external vs. internal libraries.

### `melodies_monet/driver/model.py`
| Function | Category | Reason for Categorization | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| `__init__` | `internal` | Initializes model object state. | - |
| `__repr__` | `internal` | Object string representation. | - |
| `glob_files` | `monetio` | Pre-ingestion file path expansion and tutorial data retrieval. | `glob`, `numpy`, `tutorial` |
| `open_model_files`| `monetio` | Core data ingestion delegating to the `monetio` Unified Reader API. | `monetio`, `xarray`, `util.tsub` |
| `rename_vars` | `internal` | Sanitizes variable names according to user configuration. | `xarray` |
| `mask_and_scale` | `internal` | Applies user-defined scaling and masking to ingested data. | `xarray` |
| `sum_variables` | `internal` | Logic for deriving new variables through summation. | `xarray` |

### `melodies_monet/driver/observation.py`
| Function | Category | Reason for Categorization | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| `__init__` | `internal` | Initializes observation object state. | - |
| `__repr__` | `internal` | Object string representation. | - |
| `open_obs` | `monetio` | Core observation ingestion via `monetio` Unified Reader API. | `monetio`, `xarray`, `util.read_util`|
| `add_coordinates_ground`| `internal` | Internal logic for adding coordinates to stationary ground data. | `xarray`, `numpy` |
| `rename_vars` | `internal` | Internal variable renaming logic for observations. | `xarray` |
| `open_sat_obs` | `monetio` | Satellite-specific ingestion using `monetio`'s satellite readers. | `monetio`, `util.tsub` |
| `filter_obs` | `internal` | Applies user-defined filters to observational datasets. | `xarray` |
| `mask_and_scale` | `internal` | Internal logic for applying masks and scaling (unit conversion). | `xarray` |
| `sum_variables` | `internal` | Derives new observational variables through summation. | `xarray` |
| `resample_data` | `internal` | Temporal resampling logic using `xarray`. | `xarray` |
| `obs_to_df` | `internal` | Conversion to pandas DataFrame for use in the pairing engine. | `pandas`, `xarray` |

### `melodies_monet/driver/pair.py`
| Function | Category | Reason for Categorization | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| `__init__` | `internal` | Initializes pair object state. | - |
| `__repr__` | `internal` | Object string representation. | - |
| `fix_paired_xarray`| `monet` | Standardizes and reformats paired datasets for the MM framework. | `xarray`, `pandas` |

### `melodies_monet/driver/analysis.py`
| Function | Category | Reason for Categorization | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| `__init__` | `internal` | Initializes the orchestrator class state. | - |
| `__repr__` | `internal` | Object string representation. | - |
| `read_control` | `internal` | Orchestrator logic for parsing the YAML configuration. | `yaml`, `pandas`, `dask` |
| `save_analysis` | `internal` | Orchestrator logic for persisting analysis state to disk. | `util.write_util` |
| `read_analysis` | `internal` | Orchestrator logic for loading previously saved state. | `util.read_util` |
| `setup_regridders` | `monet` | Initializes spatial regridding via `monet`-compatible utilities. | `util.regrid_util` |
| `open_models` | `monetio` | Orchestrates discovery and ingestion of model datasets. | `driver.model`, `cartopy` |
| `open_obs` | `monetio` | Orchestrates discovery and ingestion of observation datasets. | `driver.observation` |
| `setup_obs_grid` | `monet` | Defines a uniform spatial grid for gridded analysis. | `util.grid_util`, `xarray` |
| `update_obs_gridded_data`| `monet` | Maps discrete points to the spatial grid. | `util.grid_util`, `pandas` |
| `normalize_obs_gridded_data`| `monet` | Averages gridded data by counts. | `util.grid_util`, `xarray` |
| `pair_data` | `monet` | The core orchestrator for spatial/temporal pairing. | `monet`, `util.tools`, `util.sutil` |
| `concat_pairs` | `internal` | Placeholder for multi-interval pair concatenation. | - |
| `plotting` | `monet-plots`| Orchestrator for all visualization tasks. | `melodies_monet.plots`, `matplotlib` |
| `stats` | `monet-stats`| Orchestrator for metric calculations. | `melodies_monet.stats`, `pandas` |

## 2. Abstraction Recommendations & Hard-Coded Elements

During the analysis, the following hard-coded paths, thresholds, and magic numbers were identified. These should be abstracted into the YAML orchestrator to improve scientific flexibility.

### General Configuration
- **Control Filename:** `control.yaml` is hard-coded as the default in `analysis.__init__`.
- **Radius of Influence:** `1e6` (meters) is a hard-coded default in multiple places (`open_models`, `pair.__init__`).

### Sampling & Interpolation
- **Overpass Times:** Satellite overpass times are currently hard-coded (e.g., TROPOMI at 13:30 local, MOPITT at 10:30 local). These should be configurable parameters for sampling.
- **Vertical Resolution:** For curtain plots, the vertical pressure interval (`10,000 Pa`) and interpolation levels (`100`) are fixed.
- **Vertical Coordinate:** The framework assumes an altitude/vertical dimension named exactly `'z'`. This requirement should be made explicit or configurable.

### File I/O & Variables
- **Variable Exclusions:** CMAQ variables like `'temperature_k'` and `'pres_pa_mid'` are hard-coded to be removed from ingestion lists.
- **Output Naming:** The paired file naming convention (`"{}_{}.nc".format(p.obs, p.model)`) is fixed in `pair_data`.

## 3. Mapping Methodology

The categorization follows the **Conductor Architecture**:
- **`monetio`**: Handles the mapping from external data formats (NetCDF, ICARTT, CSV) to standardized `xarray` or `pandas` objects.
- **`monet`**: Performs spatial and temporal alignment (Pairing Engine) and coordinate transformations.
- **`monet-stats`**: Computes mathematical and scientific metrics on the aligned data.
- **`monet-plots`**: Renders scientific visualizations and saves them to disk.
- **`internal`**: Orchestration logic that manages the overall flow, state, and user-defined configuration.
