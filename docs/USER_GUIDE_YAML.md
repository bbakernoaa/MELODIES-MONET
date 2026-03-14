# Comprehensive User Guide: The MELODIES-MONET Orchestrator and Control YAML

MELODIES-MONET features a reorganized, unified control structure designed for clarity, scalability, and high-performance execution. This guide explains the architecture of the new control file and how to leverage its features for scientific evaluation.

---

## 1. Core Philosophy: The Workflow Graph

The orchestrator operates on a Directed Acyclic Graph (DAG). Every execution follows a standard path:
1.  **Data Nodes**: Load and preprocess model or observation data.
2.  **Pairing Nodes**: Spatial and temporal alignment between an "Experiment" (model) and a "Reference" (observation or another model).
3.  **Analysis Nodes**: Calculation of scientific metrics and generation of visualizations.

By decoupling data definition from analysis tasks, you can define a data source once and use it in multiple evaluations or plots without redundant loading.

---

## 2. Global Orchestration (`analysis`)

The `analysis` section configures the orchestrator's behavior and the execution environment.

### Execution Modes
- **Sequential**: (Default) Runs tasks one-by-one in topological order.
- **Prefect/Dask**: (Enabled via `use_prefect: True`) Parallelizes tasks across a local machine or an HPC cluster using a distributed scheduler.

### Platform Support
MELODIES-MONET includes built-in support for major NOAA (Hera, Jet, Orion, Gaea) and NCAR (Casper, Derecho) platforms. Hostnames are auto-detected to apply optimal SLURM/PBS settings.

```yaml
analysis:
  start_time: "2019-09-01 00:00:00"
  end_time: "2019-09-07 00:00:00"
  output_dir: "./output"
  use_prefect: True
  time_interval: "1D" # Chunks execution by day to save memory
  platform: "hera"    # Force a specific platform config
  account: "your_account"
```

---

## 3. Unified Data Definition (`data`)

All inputs are defined under the `data` key. MELODIES-MONET uses the `type` field to distinguish between models and observations.

### Data Loading and Reader Kwargs
You can pass additional keyword arguments directly to the underlying `monetio.load` function for each data source using `mod_kwargs` or `obs_kwargs`. This allows for fine-grained control over reader-specific options.

```yaml
data:
  my_model:
    type: model
    source: 'cmaq'
    files: '/path/to/files/*.nc'
    mod_kwargs:
      preprocess: True      # Example monetio reader kwarg
      engine: 'netcdf4'    # Example xarray engine kwarg
    variables:
      O3: {rename: 'OZONE', unit_scale: 1.0}
```

### Data Processing Capabilities
Within each data entry, you can perform standard transformations:
- **`variables`**: Rename variables, apply unit scaling (multiply, divide, add, subtract), and set detection limits (`LLOD`).
- **`variable_summing`**: Create new variables by summing existing ones (e.g., `PM25 = EC + OC + ...`).
- **`data_proc`**: Apply filters to observations (e.g., `isin`, `==`, `>=`, `!=`).
- **`resample`**: Resample time-series data using standard rules (e.g., '1H', 'D').

### Plotting Defaults
The `plot_kwargs` key allows you to define the "brand" of a data source. These settings (color, marker, label) will follow this data source into every plot it appears in.

---

## 4. Logical Pairings (`evaluations`)

The `evaluations` section defines the relational edges. It tells the orchestrator to pair an `exp` (Experiment/Model) with a `ref` (Reference/Observation).

```yaml
evaluations:
  o3_evaluation:
    exp: 'my_model'
    ref: 'airnow'
    mapping:
      OZONE: 'OZONE' # model_var: obs_var
```

---

## 5. Analysis Tasks (`stats` and `plotting`)

Tasks are performed on the results of the pairings defined in `evaluations`.

### Dynamic Statistics (`stats`)
MELODIES-MONET dynamically looks up any metric available in the `monet-stats` library. If a metric requires a threshold (like HSS or POD), simply include it in the group configuration.

```yaml
stats:
  summary:
    type: table
    data: ['o3_evaluation']
    stat_list: ['MB', 'RMSE', 'KGE', 'HSS']
    threshold: 70.0 # Ozone exceedance threshold
```

### Advanced Plotting (`plotting`)
Plotting groups map to `monet-plots` classes. Most common matplotlib and cartopy keyword arguments are supported and passed through.

```yaml
plotting:
  ozone_ts:
    type: timeseries
    data: ['o3_evaluation']
    title: "Ozone Evaluation"
    plot_kwargs: {linewidth: 2.0} # Group-level override
    figsize: [12, 6]
```

---

## 6. Optimization: Saving and Reading

For intensive pairings, you can save intermediate "Paired" objects.

```yaml
analysis:
  save:
    paired: {method: 'netcdf', prefix: 'my_study'}
```

Subsequent runs can use `read` to bypass the pairing step and go straight to analysis:
```yaml
analysis:
  read:
    paired: {method: 'netcdf', prefix: 'my_study'}
```

---

## 7. Performance and Scalability

### Lazy-First Architecture
MELODIES-MONET is designed for "Big Data" using Xarray and Dask.
- **Strict Lazy Preservation**: Data loading and pairing stay lazy. Computation is delayed until final output, ensuring efficient memory usage.
- **Task Routing**: Use `use_dtn: True` in the `data` section to route heavy data loading tasks to Data Transfer Nodes (DTN).
- **Scalable HPC**: The orchestrator can automatically provision and scale Dask-Jobqueue clusters on SLURM and PBS systems.
