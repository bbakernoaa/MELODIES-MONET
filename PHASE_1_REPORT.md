# Phase 1: Scoping & Data Contracts Report

## 1. Functional Audit

This audit categorizes the internal functions and classes of MELODIES-MONET into their future homes in the MONET ecosystem.

### Targeted Repository Branches
- **monetio**: [develop](https://github.com/bbakernoaa/monetio/tree/develop)
- **monet**: [feature/interp_improvements](https://github.com/bbakernoaa/monet/tree/feature/interp_improvements)
- **monet-stats**: [dev](https://github.com/bbakernoaa/monet-stats/tree/dev)
- **monet-plots**: [main](https://github.com/bbakernoaa/monet-plots/tree/main)

### 1.1: Data Ingestion & Standardization (monetio)
*Responsible for fetching, reading, and metadata standardization.*

| Function / Module | Description |
| :--- | :--- |
| Consolidated to `monetio` / `orchestrator` | `read_aircraft_obs_csv` moved to `monetio.load`; `read_pkl`/`read_analysis_ncf` handled by orchestrator. |
| `melodies_monet/util/read_grid_util.py` | Functions `read_grid_models` and `read_grid_obs`. |
| `melodies_monet/util/time_interval_subset.py` | Time-based file filtering for various satellite and model products. |
| `melodies_monet/util/write_util.py` | NetCDF/Zarr/Pickle persistence logic (`write_analysis_ncf`, `write_ncf`). |
| `melodies_monet/driver/model.py` | `glob_files`, `open_model_files` (logic delegating to `monetio`). |
| `melodies_monet/driver/observation.py` | `open_obs`, `open_sat_obs` (logic delegating to `monetio`). |

### 1.2: Spatial/Temporal Pairing & Interpolation (monet)
*Responsible for regridding, vertical/horizontal interpolation, and pairing logic.*

| Function / Module | Description |
| :--- | :--- |
| `melodies_monet/util/regrid_util.py` | `setup_regridder` (xesmf integration). |
| `melodies_monet/util/grid_util.py` | Sparse and uniform grid generation/updates. |
| `melodies_monet/util/region_select.py` | Masking and spatial selection (`select_region`, `get_regions`). |
| `melodies_monet/util/satellite_utilities.py` | `vertical_regrid`, `space_and_time_pairing`, `mod_to_overpasstime`. |
| `melodies_monet/util/sat_l2_swath_utility*.py` | Swath-to-grid interpolation and weighting logic. |
| `melodies_monet/util/tools.py` | `vert_interp`, `mobile_and_ground_pair`, `resample_stratify`. |
| `melodies_monet/driver/pair.py` | High-level pairing orchestration and `fix_paired_xarray`. |

### 1.3: Statistical Metrics (monet-stats)
*Responsible for RMSE, Bias, and other scientific metrics.*

| Function / Module | Description |
| :--- | :--- |
| `melodies_monet/stats/proc_stats.py` | `calc` function and `produce_stat_dict`. |
| `melodies_monet/util/tools.py` | Statistical helpers: `linregress`, `calc_24hr_ave`, `calc_8hr_rolling_max`. |

### 1.4: Visualization (monet-plots)
*Responsible for all Matplotlib and Cartopy based plotting.*

| Function / Module | Description |
| :--- | :--- |
| `melodies_monet/plots/surfplots.py` | Surface site time series, spatial bias, and overlay plots. |
| `melodies_monet/plots/aircraftplots.py` | Curtain plots, vertical profiles, and aircraft-specific viz. |
| `melodies_monet/plots/satplots.py` | Satellite-specific visualizations. |
| `melodies_monet/plots/sonde_plots.py` | Ozonesonde vertical plots. |
| `melodies_monet/plots/xarray_plots.py` | Generic Xarray-based plotting utilities. |
| `melodies_monet/plots/Plot_2D.py` | 2D spatial plotting class. |

---

## 2. The "Paired Data" Contract

To ensure `monet-stats` and `monet-plots` can operate as "pure consumers," a standardized intermediate format is required.

### 2.1: Format Specification
- **Primary Format**: NetCDF4 (with groups) or Zarr.
- **Data Structure**: Xarray Dataset.
- **Standards**:
    - **Point Data**: Utilize the **UGRID** standard (via `monetio`) for representation of discrete sampling geometries.
    - **Model-to-Model / Model-to-Satellite**: Data should be maintained in the **Reference Grid** (typically the model's native grid or the satellite's retrieval grid) to preserve spatial integrity and avoid unnecessary re-interpolation.

### 2.2: Mandatory Content
A "Paired Data" file must contain:

1.  **Coordinates**:
    - `time`: Primary time dimension (UTC).
    - `x` (or `site`): Primary spatial dimension for point data.
    - `lat` / `lon`: Latitude and Longitude for every data point.
    - `z` / `pressure`: Vertical coordinate if applicable.

2.  **Variables**:
    - `[var_name]_obs`: The observed values.
    - `[var_name]_model`: The paired model values.
    - Suffixes should be standardized (e.g., `_obs` and `_mod`).

3.  **Metadata (Attributes)**:
    - `units`: Mandatory for both obs and model variables.
    - `long_name`: Descriptive name for plotting labels.
    - `history`: Lineage of the pairing process.

4.  **Site Metadata** (for point-based data):
    - `siteid` / `x`: Unique identifier and primary spatial index.
    - Standardized Site Variables: `latitude`, `longitude`, `site`, `msa_code`, `cmsa_name`, `epa_region`, `state_name`, `msa_name`, `utcoffset`.
    - These are included as non-dimension coordinates or auxiliary variables (time-independent) to enable grouping and regional analysis without re-loading original metadata.

### 2.3: Benefit
This contract allows the Orchestrator to persist the state after the `pair` node, enabling the `stats` and `plots` nodes to be executed on different compute resources or at a later time without access to the massive original model/obs input files.
