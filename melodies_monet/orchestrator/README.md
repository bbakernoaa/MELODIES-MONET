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
    cluster_type: "slurm"  # options: local, slurm, pbs, lsf, fargate
    cluster_kwargs:
      nodes: 2
      cores: 24
      memory: "128GB"
      walltime: "02:00:00"
```

### HPC Support

The `ClusterFactory` automatically detects project/account codes from environment variables:
- `PROJECT` or `ACCOUNT` for SLURM and PBS.
- `PROJECT` for LSF.

## Tasks

- `run_data_task`: Handles loading of model and observation data using `monetio`.
- `run_pairing_task`: Performs spatial/temporal pairing using `monet.pair`. It uses the `worker_client` pattern for fine-grained parallelism.
- `run_stats_task`: Calculates statistical metrics.
- `run_plotting_task`: Generates visualizations.
