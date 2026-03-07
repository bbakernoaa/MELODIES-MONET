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

## Dask and HPC Performance

By using the `run-node` command, each task in the workflow inherits the Orchestrator's ability to initialize a Dask cluster. This ensures that memory-intensive operations (like pairing large Xarray datasets) are automatically distributed across the allocated HPC resources, leveraging the backend-agnostic performance of MELODIES-MONET.
