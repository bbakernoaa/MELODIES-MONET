from prefect import flow, get_run_logger, task
from prefect_dask import DaskTaskRunner
import dask
from melodies_monet.orchestrator.tasks import run_data_task, run_pairing_task, run_stats_task, run_plotting_task
from melodies_monet.orchestrator.cluster_config import ClusterFactory, PLATFORM_CONFIGS

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

        if node_type == "data":
            data_type = node_attr["data_type"]
            cfg = control_dict.get("data", {}).get(label, {})

            # Route to DTN if configured
            resources = {}
            if cfg.get("use_dtn", False):
                resources = {"dtn": 1}

            with dask.annotate(resources=resources):
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
            cfg = control_dict.get("evaluations", {}).get(label, {})
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
            cfg = control_dict.get("stats", {}).get(label, {})
            future = run_stats_task.submit(
                group_label=label,
                cfg=cfg,
                paired_dict=paired_data,
                wait_for=wait_for
            )
            task_results[node_id] = future

        elif node_type == "plot":
            cfg = control_dict.get("plotting", {}).get(label, {})
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
    logger = get_run_logger()

    analysis_cfg = orchestrator_inst.control_dict.get("analysis", {})
    dask_cfg = analysis_cfg.get("dask", {})
    platform = orchestrator_inst.platform or dask_cfg.get("platform")
    account = orchestrator_inst.account or dask_cfg.get("account")
    project = orchestrator_inst.project or dask_cfg.get("project")

    cluster_type = dask_cfg.get("cluster_type")
    cluster_kwargs = dask_cfg.get("cluster_kwargs", {})
    adaptive_kwargs = dask_cfg.get("adaptive_kwargs", {})

    logger.info(f"Provisioning cluster (platform={platform})")
    cluster = ClusterFactory.get_cluster(
        cluster_type=cluster_type,
        cluster_kwargs=cluster_kwargs,
        adaptive_kwargs=adaptive_kwargs,
        platform=platform,
        account=account,
        project=project
    )

    # Scale worker groups
    if hasattr(cluster, "worker_spec"):
        # Default scaling
        default_scale = dask_cfg.get("scale", 2)

        # Scaling overrides from worker_groups
        worker_groups = dask_cfg.get("worker_groups", {})

        for name in cluster.worker_spec:
            group_cfg = worker_groups.get(name, {})
            if "adaptive_kwargs" in group_cfg:
                cluster.adapt(name=name, **group_cfg["adaptive_kwargs"])
            elif "scale" in group_cfg:
                cluster.scale(group_cfg["scale"], name=name)
            else:
                # Platform/Default fallback
                if name == "dtn":
                    cluster.scale(1, name=name)
                else:
                    if adaptive_kwargs:
                        cluster.adapt(name=name, **adaptive_kwargs)
                    else:
                        cluster.scale(default_scale, name=name)

    task_runner = DaskTaskRunner(address=cluster.scheduler_address)

    try:
        intervals = orchestrator_inst.time_intervals if orchestrator_inst.time_intervals else [None]
        all_results = []
        for interval in intervals:
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
