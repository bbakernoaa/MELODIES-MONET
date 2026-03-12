# SPDX-License-Identifier: Apache-2.0
#
"""
Prefect flow engine for MELODIES-MONET.
"""

import networkx as nx
from prefect import flow, task, get_run_logger
from dask.distributed import Client, worker_client
from melodies_monet.orchestrator.cluster_config import ClusterFactory


@task
def run_mm_node(node_name, func, *args, **kwargs):
    """
    Prefect task to run a single node from the MM Orchestrator's DAG.

    IMPORTANT: All MM tasks MUST remain backend-agnostic and preserve Dask laziness.
    Never use .compute(), .load(), or .values inside functions being executed.
    """
    logger = get_run_logger()
    logger.info(f"Starting MM task: {node_name}")

    # For tasks that could benefit from spawning their own sub-tasks on the cluster
    # (e.g., heavy pairing or stats calculation), use the worker_client context.
    try:
        with worker_client():
            logger.info(f"Task {node_name} connected to cluster via worker_client.")
            result = func(*args, **kwargs)
    except (ValueError, RuntimeError):
        # Fallback if not running within a dask worker context
        logger.info(f"Task {node_name} running in local context.")
        result = func(*args, **kwargs)

    logger.info(f"Completed MM task: {node_name}")
    return result


@flow(name="MM Execution Flow")
def execute_dag_flow(orchestrator):
    """
    Inner flow that maps Orchestrator DAG (NetworkX) to Prefect Tasks.
    """
    logger = get_run_logger()
    task_futures = {}

    # Traverse the graph in topological order to ensure dependencies are met
    for node in nx.topological_sort(orchestrator.graph):
        node_data = orchestrator.graph.nodes[node]
        func = node_data["func"]

        # Extract args/kwargs if they exist in the node data
        node_args = node_data.get("args", [])
        node_kwargs = node_data.get("kwargs", {})

        # Determine dependencies (predecessors in the DAG)
        dependencies = [
            task_futures[dep] for dep in orchestrator.graph.predecessors(node)
        ]

        # Submit the task to Prefect.
        # Using wait_for to manage dependencies as per the NetworkX structure.
        future = run_mm_node.submit(
            node_name=node, func=func, *node_args, **node_kwargs, wait_for=dependencies
        )
        task_futures[node] = future

    logger.info("All tasks submitted to Prefect.")

    # Wait for all tasks to finish
    for node, future in task_futures.items():
        future.wait()

    return task_futures


@flow(name="MELODIES-MONET Workflow")
def mm_prefect_flow(orchestrator):
    """
    Wrapper flow that provisions a Dask cluster and launches the execution flow.
    """
    logger = get_run_logger()

    # 1. Initialize Dask Cluster from Config
    dask_config = orchestrator.ana.control_dict.get("analysis", {}).get("dask", {})
    cluster = None
    client = None
    task_runner = None

    if dask_config:
        cluster = ClusterFactory.create_cluster(dask_config)
        client = Client(cluster)
        logger.info(f"Dask cluster initialized: {cluster.dashboard_link}")

        # Configure the DaskTaskRunner to use our new cluster
        try:
            from prefect_dask import DaskTaskRunner

            task_runner = DaskTaskRunner(address=cluster.scheduler_address)
            logger.info(
                f"Prefect DaskTaskRunner configured with address: {cluster.scheduler_address}"
            )
        except ImportError:
            logger.warning(
                "prefect-dask is not installed. Tasks will run on the default runner."
            )
    else:
        logger.info("No Dask configuration found. Running in local mode.")

    try:
        # 2. Execute the DAG within a flow configured with the DaskTaskRunner
        if task_runner:
            result = execute_dag_flow.with_options(task_runner=task_runner)(
                orchestrator
            )
        else:
            result = execute_dag_flow(orchestrator)

        return result

    finally:
        # 3. Clean up resources
        if client:
            client.close()
        if cluster:
            cluster.close()
            logger.info("Dask cluster closed.")
