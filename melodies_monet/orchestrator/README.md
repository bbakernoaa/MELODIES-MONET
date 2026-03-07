# MELODIES-MONET Prefect Orchestrator

This module provides a Prefect-based orchestration layer for MELODIES-MONET, enabling scalable, distributed execution of verification workflows across Local, HPC, and Cloud environments.

## Overview

The Prefect Orchestrator translates the internal MELODIES-MONET task graph (defined using NetworkX) into a Prefect workflow. It leverages Dask for distributed computing and supports dynamic cluster provisioning on various platforms.

### Key Features

- **Nested Flow Pattern**: A wrapper flow manages infrastructure (Dask cluster provisioning) while an inner execution flow handles task scheduling.
- **Elastic Cluster Management**: Integration with `dask-jobqueue` and `dask-cloud-provider` for Slurm, PBS, LSF, and AWS Fargate.
- **Native HPC Support**: Pre-configured defaults for NCAR (Casper, Derecho) and NOAA RDHPCS (Hera, Jet, Orion, Hercules, Gaea, Ursa).
- **Worker-Client Pattern**: Support for `worker_client()` allows tasks to dynamically spawn sub-tasks on the cluster for heavy computations.
- **Dependency Tracking**: Automatic translation of NetworkX DAG dependencies into Prefect's `wait_for` logic.

## Configuration

The orchestrator is configured via the `analysis: dask` section of the MELODIES-MONET control YAML.

### YAML Schema

```yaml
analysis:
  dask:
    platform: "orion"           # Optional: 'casper', 'derecho', 'hera', 'jet', 'orion', 'msu', 'hercules', 'gaea', 'ursa'
    cluster_type: "slurm"       # Optional if platform is set: 'local', 'slurm', 'pbs', 'lsf', 'fargate'
    cluster_kwargs:             # Passed to the Dask cluster constructor
      queue: "batch"
      project: "your_project"   # Automatically discovered if environment variables (PROJECT/ACCOUNT) are set
      cores: 40
      memory: "192GB"
      walltime: "02:00:00"
    adaptive:                   # Optional: Configure adaptive scaling
      enabled: true
      minimum: 1
      maximum: 20
    scale: 5                    # Optional: Static scaling (ignored if adaptive is enabled)
```

## Supported Platforms

| Platform | Scheduler | Default Cores | Default Memory | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Casper** | Slurm | 1 | 4GB | NCAR Data Analysis/Visualization |
| **Derecho** | PBS | 128 | 256GB | NCAR Supercomputer |
| **Hera** | Slurm | 40 | 92GB | NOAA RDHPCS |
| **Jet** | Slurm | 24 | 60GB | NOAA RDHPCS |
| **Orion/MSU** | Slurm | 40 | 192GB | NOAA RDHPCS |
| **Hercules** | Slurm | 80 | 512GB | NOAA RDHPCS |
| **Gaea** | Slurm | 32 | 128GB | NOAA RDHPCS |
| **Ursa** | Slurm | 44 | 192GB | NOAA RDHPCS |

## Deployment to HPC

### 1. Environment Setup

Ensure you have the necessary dependencies installed in your environment:

```bash
pip install prefect prefect-dask dask-jobqueue networkx
```

### 2. Execution

To run a workflow using the Prefect engine, you can use the internal API:

```python
from melodies_monet.driver.orchestrator import Orchestrator
from melodies_monet.orchestrator.prefect_engine import mm_prefect_flow

# Initialize MM Orchestrator
orch = Orchestrator("control.yaml")

# Launch Prefect Flow
state = mm_prefect_flow(orch)
```

### 3. Monitoring

If you have a Prefect server running, you can monitor the progress through the Prefect UI. Additionally, the Dask dashboard link will be logged, allowing you to monitor distributed task execution in real-time.

## The Worker-Client Pattern

For computationally intensive tasks (like heavy pairing or complex statistics), the orchestrator uses the Dask `worker_client()` context manager. This allows a primary Prefect task to act as a Dask client itself, spawning fine-grained sub-tasks directly onto the Dask workers.

**Important**: When implementing new driver functions, avoid "lazy breakers" such as `.compute()`, `.load()`, or `.values` inside processing logic to maintain the performance benefits of Dask's lazy execution.

## Troubleshooting

- **Project/Account Discovery**: On most HPC systems, the orchestrator will try to automatically find your project code by checking environment variables like `$PROJECT`, `$ACCOUNT`, or `$PBS_ACCOUNT`. If this fails, specify it explicitly in `cluster_kwargs`.
- **Prefect-Dask**: If `prefect-dask` is not installed, the orchestrator will log a warning and fall back to the default Prefect runner, although Dask operations will still execute on the cluster.
- **Database Locks**: If running multiple flows locally with a SQLite backend, you may encounter "database is locked" errors. Consider using a PostgreSQL backend for production Prefect deployments.
