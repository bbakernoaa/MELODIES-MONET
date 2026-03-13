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

## Multi-Node Execution

MELODIES-MONET leverages Xarray and Dask to enable tasks to utilize multiple HPC nodes.

1. **Lazy Loading**: Data is loaded lazily as Dask arrays. Large datasets are automatically partitioned across the cluster.
2. **Worker-Client Pattern**: Heavy compute tasks like `run_pairing_task` use the `worker_client` pattern to spawn sub-tasks. These sub-tasks are distributed across all available workers in the cluster, which can span many physical HPC nodes.
3. **Scaling**: You can request multiple nodes by setting `scale` or `adaptive_kwargs` in the control YAML. Each "worker" typically corresponds to one HPC job (which can be configured to use one or more nodes via `cluster_kwargs`).
