# MELODIES-MONET Prefect Orchestrator

This subpackage implements the orchestration layer for MELODIES-MONET using Prefect and Dask.

## Architecture

The orchestrator uses a nested flow pattern:
1. **`main_orchestration_flow`**: Provisions the Dask cluster based on configuration and initializes the `DaskTaskRunner`.
2. **`execute_dag_flow`**: Traverses the internal NetworkX DAG in topological order and submits tasks to the Dask cluster.

## Configuration

To enable Prefect orchestration, add the following to your control YAML:

```yaml
analysis:
  use_prefect: True
  dask:
    platform: "hera"  # optional: hera, jet, orion, hercules, gaea, casper, derecho
    # If platform is set, cluster_type and kwargs are defaulted based on predefined setups.
    cluster_kwargs:
      nodes: 4
```

### Supported Platforms

Predefined configurations are available for:
- **NOAA RDHPCS**: `hera`, `jet`, `orion`, `hercules`, `gaea`, `ursa`
- **NCAR**: `casper`, `derecho`

The `ClusterFactory` attempts to automatically detect the platform based on the hostname if not explicitly provided.

### HPC Support

The `ClusterFactory` automatically detects project/account codes from environment variables:
- `PROJECT` or `ACCOUNT` for SLURM and PBS.
- `PROJECT` for LSF.

## Tasks

- `run_data_task`: Handles loading of model and observation data using `monetio`.
- `run_pairing_task`: Performs spatial/temporal pairing using `monet.pair`. It uses the `worker_client` pattern for fine-grained parallelism.
- `run_stats_task`: Calculates statistical metrics.
- `run_plotting_task`: Generates visualizations.
