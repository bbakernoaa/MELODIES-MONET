# MELODIES-MONET Driver

The driver is the core orchestration layer of MELODIES-MONET, responsible for reading configurations, loading data, pairing models with observations, and driving statistics and plotting.

## Control YAML Schema

The control YAML uses a modular four-tier hierarchical schema designed to be concise and intuitive.

### 1. `analysis`
Global settings for the evaluation.
- `start_time` / `end_time`: Time window for the analysis.
- `output_dir`: Directory for output files.
- `debug`: Enable verbose logging.

### 2. `data`
Unified section for defining all data objects (models and observations).
Each entry uses a `type` key to categorize the data:
- `type: exp`: Experimental or model data (formerly `model`).
- `type: ref`: Reference or observational data (formerly `obs`).

Example:
```yaml
data:
  cmaq_expt:
    type: exp
    source: 'cmaq'
    files: '/path/to/cmaq/*.nc'
  airnow:
    type: ref
    source: 'airnow'
    files: '/path/to/airnow.nc'
    obs_type: pt_sfc
```
*Note: `source` is preferred over `mod_type`/`obs_type`, and `files` is preferred over `filename`.*

### 3. `evaluations`
Relational logic that defines how data objects are paired.
- `exp`: Key of the experimental data object.
- `ref`: Key of the reference data object.
- `mapping`: Species mapping between the two objects.

Example:
```yaml
evaluations:
  my_eval:
    exp: cmaq_expt
    ref: airnow
    mapping:
      O3: 'OZONE'
```

### 4. `plotting` and `stats`
Task-based sections for visualization and statistical analysis. They reference the evaluation labels defined in the `evaluations` section.

Example:
```yaml
plotting:
  plot1:
    type: 'timeseries'
    data: ['my_eval']
```

## Backward Compatibility

The driver maintains full backward compatibility for legacy control files. It transparently migrates the following at runtime:
- `model` or `models` sections are moved to `data` (with `type: model`).
- `obs` section is moved to `data` (with `type: obs`).
- `plots` section is renamed to `plotting`.
- Inline mappings within the `model` section are automatically converted to `evaluations`.

## Core Classes

- **`analysis`**: The high-level orchestrator.
- **`Data`**: Unified class for both models and observations.
- **`pair`**: Handles the spatial and temporal alignment of data objects.
