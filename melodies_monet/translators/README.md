# MELODIES-MONET Workflow Translators

This directory contains the translation layer that converts the internal **NetworkX Directed Acyclic Graph (DAG)** into executable workflow manager scripts for **ecFlow** and **Snakemake**.

## How It Works

The MELODIES-MONET `Orchestrator` builds a task graph using NetworkX. Each node represents a high-level function (e.g., `open_models`, `pair_data`, `stats`), and each edge represents a dependency.

### 1. NetworkX to Workflow Mapping

| Feature | NetworkX | ecFlow | Snakemake |
| --- | --- | --- | --- |
| **Task** | Node | `ecflow.Task` | `rule` |
| **Dependency** | Edge | `task.add_trigger()` | `rule input:` |
| **Parameters** | Node Attributes | `edit VARIABLE 'VALUE'` | `params: key="value"` |
| **Execution** | Function Reference | `RUN_COMMAND` variable | `shell:` command |

### 2. The `run-node` CLI Command

To enable distributed execution on HPC systems, the workflow managers need a way to trigger specific parts of the Orchestrator DAG. This is achieved through the `melodies-monet run-node` command:

```bash
melodies-monet run-node <control_file> <node_name>
```

When this command is executed:
1. The `Orchestrator` is initialized with the provided control file.
2. If Dask is configured in the `analysis` section, a distributed **Dask Client** is started.
3. The specific function associated with `<node_name>` is executed.

### 3. ecFlow Exporter (`ecflow_exporter.py`)

The `EcflowExporter` uses the `ecflow` Python API to generate a suite definition (`.def`).
- **Multiple Dependencies**: If a node has multiple predecessors, they are joined with ` AND ` in the trigger expression (e.g., `task.add_trigger("./task1 == complete AND ./task2 == complete")`).
- **HPC Resources**: Resource requests (e.g., `profile: high-mem`) are exported as ecFlow variables, which can be used by ecFlow's submission scripts to generate scheduler headers.

### 4. Snakemake Exporter (`snakemake_exporter.py`)

The `SnakemakeExporter` uses Jinja2 templates to generate a `Snakefile`.
- **Target Tracking**: Since Snakemake is file-driven, each rule creates a `.complete` sentinel file upon successful execution.
- **Rule Generation**: Each node in the DAG becomes a rule. Dependencies are enforced by listing the `.complete` files of predecessors as inputs.

### 5. HPC Resource Mapping (`resources.py`)

The `resources.py` module, backed by `machines.yaml`, translates abstract resource profiles into specific scheduler directives for **Slurm** and **PBS**. Supported machines include:
- **NCAR**: Casper
- **NOAA**: Hera, Jet, Gaea, Orion (msu), Ursa, Hercules

Profiles:
- `standard`: Balanced CPU and memory.
- `transfer`: Targeted for data staging on service or DTN partitions.
- `high-mem`: High memory allocation for large grid processing or intensive stats/plotting.

## Dask and Data Handoff

A common question is how Dask objects (which are in-memory references to potentially massive datasets) are "passed" between workflow managers like ecFlow or Snakemake.

### 1. File-Based Persistence

In a distributed workflow, memory is not shared between tasks running on different compute nodes. Therefore, **Dask objects are not passed directly.** Instead:
- Each task (node) in the DAG is responsible for **persisting its results to disk** (typically NetCDF or Zarr format).
- The subsequent task **re-opens these files**, often using `open_mfdataset` to maintain Dask laziness.
- This creates a **Data Contract** between nodes: the output file naming convention must match what the next node expects to read.

### 2. Distributed Dask Clients

When `melodies-monet run-node` is called:
1. It initializes a **new Dask Client** specific to that task's execution environment.
2. If the task is running on an HPC compute node (allocated via Slurm/PBS), the Dask Client can be configured to use the local resources (CPUs/Memory) assigned to that node.
3. This ensures that while the workflow manager coordinates the *sequence* of tasks, Dask coordinates the *parallelism within* each task.

### 3. Workflow vs. Dask Parallelism

- **Workflow Manager (ecFlow/Snakemake)**: Handles **Inter-task parallelism** (e.g., running `open_models` and `open_obs` at the same time).
- **Dask**: Handles **Intra-task parallelism** (e.g., distributing the actual computation of a 100GB regridding operation across 40 cores).

By combining these, MELODIES-MONET achieves massive scalability: the workflow manager manages the cluster's queue and task dependencies, while Dask manages the data-local compute performance.

## Dynamic Scaling with `dask-jobqueue`

For tasks that require more resources than a single compute node can provide, MELODIES-MONET supports **dask-jobqueue**. This allows a single workflow node to dynamically spawn its own "helper" compute nodes.

### Configuration

Add a `jobqueue` section to your `dask` configuration in the control YAML:

```yaml
analysis:
  dask:
    jobqueue:
      type: "slurm"  # pbs, lsf
      kwargs:
        nodes: 1
        cores: 40
        memory: "128GB"
        walltime: "01:00:00"
        interface: "ib0"
      scale:
        jobs: 4  # Static scaling: spawn 4 helper jobs
      # OR adaptive scaling
      # scale:
      #   adaptive:
      #     minimum_jobs: 1
      #     maximum_jobs: 10
```

When `run-node` is executed with this configuration:
1. It initializes a Cluster object (e.g., `SLURMCluster`) using the provided `kwargs`.
2. It scales the cluster (statically or adaptively).
3. It connects the Dask Client to this dynamic cluster.
4. After the task completes, the dynamic nodes are automatically released.
