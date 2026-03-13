import sys
from unittest.mock import MagicMock, patch

# Mock heavy dependencies
sys.modules["monet"] = MagicMock()
sys.modules["monetio"] = MagicMock()
sys.modules["monet_plots"] = MagicMock()
sys.modules["monet_stats"] = MagicMock()
sys.modules["xesmf"] = MagicMock()
sys.modules["cartopy"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["prefect"] = MagicMock()
sys.modules["prefect_dask"] = MagicMock()
sys.modules["dask.distributed"] = MagicMock()

from melodies_monet.driver.orchestrator import orchestrator


@patch("melodies_monet.orchestrator.flows.main_orchestration_flow")
def test_orchestrator_run_prefect(mock_flow):
    """Test that orchestrator.run() calls the Prefect flow when enabled."""
    an = orchestrator()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-01",
            "end_time": "2019-08-02",
            "output_dir": ".",
            "use_prefect": True,
            "platform": "hera",
        }
    }

    an.graph_engine.get_execution_order = MagicMock(return_value=[])
    an.run()
    assert mock_flow.called
    assert an.platform == "hera"


def test_cluster_factory_platform_merge():
    """Test that ClusterFactory correctly merges platform defaults."""
    from melodies_monet.orchestrator.cluster_config import ClusterFactory

    with patch("dask_jobqueue.SLURMCluster") as mock_slurm:
        ClusterFactory.get_cluster(platform="hera", account="test_acc")
        args, kwargs = mock_slurm.call_args
        assert kwargs["partition"] == "batch"
        assert kwargs["account"] == "test_acc"
        assert kwargs["cores"] == 40


@patch("dask.annotate")
def test_task_routing_resources(mock_annotate):
    """Test that tasks are routed with correct resources in the flow."""
    import networkx as nx

    from melodies_monet.orchestrator.flows import execute_dag_flow

    graph = nx.DiGraph()
    graph.add_node("data_mod", type="data", data_type="model", label="mod1")

    # Unified data section
    control_dict = {"data": {"mod1": {"type": "model"}}, "analysis": {}}

    with patch("melodies_monet.orchestrator.tasks.run_data_task.submit"):
        execute_dag_flow(
            control_dict=control_dict,
            execution_order=["data_mod"],
            graph=graph,
            pairing_kwargs={},
        )
        assert mock_annotate.called
        args, kwargs = mock_annotate.call_args
        assert kwargs["resources"] == {"dtn": 1}
