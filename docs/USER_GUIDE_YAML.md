# Comprehensive User Guide: The MELODIES-MONET Orchestrator and Control YAML

MELODIES-MONET features a reorganized, unified control structure designed for clarity, scalability, and high-performance execution. This guide explains the architecture of the new control file and how to leverage its features for scientific evaluation.

---

## 1. Core Philosophy: The Workflow Graph

The orchestrator operates on a Directed Acyclic Graph (DAG). Every execution follows a standard path:
1.  **Data Nodes**: Load and preprocess datasets.
2.  **Pairing Nodes**: Spatial and temporal alignment between an "Experiment" (typically a model) and a "Reference" (typically an observation).
3.  **Analysis Nodes**: Calculation of scientific metrics and generation of visualizations.

By decoupling data definition from analysis tasks, you can define any dataset once and use it in multiple evaluations or plots without redundant loading.

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
```

---

## 3. Unified Data Entrance (`data`)

All datasets—whether they are models, surface observations, or satellite products—are defined under the unified `data` entrance. MELODIES-MONET treats all datasets as equal scientific objects.

### Dataset Configuration
- **`source`**: The reader name (e.g., `cmaq`, `wrfchem`, `airnow`).
- **`files`**: File path pattern or `example:ID`.
- **`kwargs`**: Pass additional keyword arguments directly to the underlying `monetio.load` function. This allows for reader-specific or xarray-specific options.

```yaml
data:
  my_dataset:
    source: 'cmaq'
    files: '/path/to/files/*.nc'
    kwargs:
      preprocess: True      # reader-specific kwarg
      engine: 'netcdf4'    # xarray engine kwarg
    variables:
      O3: {rename: 'OZONE', unit_scale: 1.0}
    plot_kwargs: {color: 'blue', label: 'CMAQ v5.4'}
```

### Data Processing Capabilities
- **`variables`**: Rename variables, apply unit scaling (multiply, divide, add, subtract), and set detection limits (`LLOD`).
- **`variable_summing`**: Create new variables by summing existing ones.
- **`data_proc`**: Apply arbitrary filters to the dataset (e.g., `isin`, `==`, `>=`, `!=`).
- **`resample`**: Resample time-series data using standard rules (e.g., '1H', 'D').

---

## 4. Logical Pairings (`evaluations`)

The `evaluations` section defines the relational edges in the graph. It tells the orchestrator to pair an `exp` (Experiment) with a `ref` (Reference).

```yaml
evaluations:
  o3_evaluation:
    exp: 'my_model'
    ref: 'airnow_obs'
    mapping:
      OZONE: 'OZONE' # experiment_var: reference_var
```

---

## 5. Analysis Tasks (`stats` and `plotting`)

### Dynamic Statistics (`stats`)
Specify any metric available in `monet-stats`. Thresholds for categorical metrics are passed directly from the group configuration.

```yaml
stats:
  summary:
    type: table
    data: ['o3_evaluation']
    stat_list: ['MB', 'RMSE', 'KGE', 'HSS']
    threshold: 70.0
```

### Advanced Plotting (`plotting`)
Plotting groups map to `monet-plots` classes. Visual preferences from `data.plot_kwargs` are merged with group-level customizations.

```yaml
plotting:
  ozone_ts:
    type: timeseries
    data: ['o3_evaluation']
    title: "Regional Ozone Evaluation"
    plot_kwargs: {linewidth: 2.0}
    figsize: [12, 6]
```

---

## 6. Performance and Scalability

### Lazy-First Architecture
MELODIES-MONET preserves Xarray/Dask laziness throughout the entire workflow. Laziness is only broken at the final rendering (plots) or output (CSV/NetCDF) steps.

- **Task Routing**: Use `use_dtn: True` in the `data` section to ensure data loading occurs on HPC Data Transfer Nodes.
- **Scalable Dask**: Clusters can be automatically provisioned and scaled on SLURM/PBS systems via the `analysis` configuration.
