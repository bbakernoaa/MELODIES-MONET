from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
from melodies_monet.orchestrator.tasks import run_data_task, run_pairing_task, run_stats_task, run_plotting_task
from melodies_monet.orchestrator.cluster_config import ClusterFactory

@flow
def execute_dag_flow(control_dict, execution_order, graph, pairing_kwargs, time_interval=None):
    logger = get_run_logger()
    logger.info(f"Starting DAG execution flow for interval: {time_interval}")

    task_results = {}
    loaded_data = {}
    paired_data = {}

    for node_id in execution_order:
        node_attr = graph.nodes[node_id]
        node_type = node_attr["type"]
        label = node_attr["label"]

        upstream_nodes = list(graph.predecessors(node_id))
        wait_for = [task_results[up_id] for up_id in upstream_nodes]

        # Determine resource tags for task routing
        # Example: routing data tasks to 'dtn' workers
        tags = []
        if node_type == "data":
            tags.append("dtn")
        else:
            tags.append("compute")

        with task.tags(*tags):
            if node_type == "data":
                data_type = node_attr["data_type"]
                cfg_section = "models" if data_type == "model" else "obs"
                cfg = control_dict[cfg_section][label]

                future = run_data_task.submit(
                    data_type=data_type,
                    label=label,
                    cfg=cfg,
                    time_interval=time_interval,
                    wait_for=wait_for
                )
                task_results[node_id] = future
                loaded_data[label] = future

            elif node_type == "pair":
                cfg = control_dict["evaluations"][label]
                mod_label = cfg.get("model", cfg.get("exp"))
                obs_label = cfg.get("obs", cfg.get("ref"))

                future = run_pairing_task.submit(
                    eval_label=label,
                    cfg=cfg,
                    model_inst=loaded_data[mod_label],
                    obs_inst=loaded_data[obs_label],
                    pairing_kwargs=pairing_kwargs,
                    wait_for=wait_for
                )
                task_results[node_id] = future
                paired_data[label] = future

            elif node_type == "stats":
                cfg = control_dict["stats"][label]
                future = run_stats_task.submit(
                    group_label=label,
                    cfg=cfg,
                    paired_dict=paired_data,
                    wait_for=wait_for
                )
                task_results[node_id] = future

            elif node_type == "plot":
                cfg = control_dict["plotting"][label]
                future = run_plotting_task.submit(
                    group_label=label,
                    cfg=cfg,
                    paired_dict=paired_data,
                    wait_for=wait_for
                )
                task_results[node_id] = future

    return task_results

@flow
def main_orchestration_flow(orchestrator_inst):
    """
    Main flow that provisions the cluster and runs the DAG subflow.
    """
    logger = get_run_logger()

    analysis_cfg = orchestrator_inst.control_dict.get("analysis", {})
    dask_cfg = analysis_cfg.get("dask", {})
    cluster_type = dask_cfg.get("cluster_type", "local")
    cluster_kwargs = dask_cfg.get("cluster_kwargs", {})
    adaptive_kwargs = dask_cfg.get("adaptive_kwargs", {})

    # Support for multiple worker groups (partitions/qos)
    worker_groups = dask_cfg.get("worker_groups", {})

    logger.info(f"Provisioning {cluster_type} cluster")
    cluster = ClusterFactory.get_cluster(cluster_type, cluster_kwargs, adaptive_kwargs)

    # Add extra worker groups to the cluster if defined
    # This assumes the Dask cluster object (e.g., SLURMCluster) supports multiple job types
    # or we manage them via the SpecCluster interface.
    if worker_groups and hasattr(cluster, "new_spec"):
        for name, spec in worker_groups.items():
            logger.info(f"Adding worker group: {name}")
            # Generate platform-specific spec refinements
            refined_spec = ClusterFactory.get_worker_spec(cluster_type, spec)
            cluster.new_spec(name, refined_spec)
            if "adaptive_kwargs" in spec:
                cluster.adapt(name=name, **spec["adaptive_kwargs"])
            elif "scale" in spec:
                cluster.scale(spec["scale"], name=name)

    task_runner = DaskTaskRunner(address=cluster.scheduler_address)

    try:
        intervals = orchestrator_inst.time_intervals if orchestrator_inst.time_intervals else [None]
        all_results = []

        for interval in intervals:
            logger.info(f"Submitting subflow for interval: {interval}")
            results = execute_dag_flow.with_options(task_runner=task_runner)(
                control_dict=orchestrator_inst.control_dict,
                execution_order=orchestrator_inst.graph_engine.get_execution_order(),
                graph=orchestrator_inst.graph_engine.graph,
                pairing_kwargs=orchestrator_inst.pairing_kwargs,
                time_interval=interval
            )
            all_results.append(results)

        return all_results
    finally:
        cluster.close()
