# SPDX-License-Identifier: Apache-2.0
#
import sys
from unittest.mock import MagicMock, patch

# Mock only what's absolutely necessary and not installed
if "monet" not in sys.modules:
    sys.modules["monet"] = MagicMock()
if "monetio" not in sys.modules:
    sys.modules["monetio"] = MagicMock()

sys.modules["xesmf"] = MagicMock()

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
    # Manually set attributes that would be set by read_control
    an.platform = "hera"

    an.graph_engine.get_execution_order = MagicMock(return_value=[])
    an.run()
    assert mock_flow.called
    assert an.platform == "hera"


def test_cluster_factory_platform_merge():
    """Test that ClusterFactory correctly merges platform defaults."""
    from melodies_monet.orchestrator.cluster_config import ClusterFactory

    # Mock SpecCluster to avoid actual job submission during tests
    with patch("dask_jobqueue.SLURMCluster"):
        with patch("dask.distributed.SpecCluster", return_value=MagicMock()) as mock_spec:
            ClusterFactory.get_cluster(platform="hera", account="test_acc")
            # When platform has worker_groups, it uses SpecCluster
            assert mock_spec.called
            args, kwargs = mock_spec.call_args
            worker_specs = args[0]
            assert "compute" in worker_specs
            assert worker_specs["compute"]["options"]["queue"] == "batch"
            assert worker_specs["compute"]["options"]["account"] == "test_acc"


@patch("dask.annotate")
def test_task_routing_resources(mock_annotate):
    """Test that tasks are routed with correct resources in the flow."""
    import networkx as nx
    from prefect.testing.utilities import prefect_test_harness

    from melodies_monet.orchestrator.flows import execute_dag_flow

    graph = nx.DiGraph()
    graph.add_node("data_mod", type="data", data_type="model", label="mod1")

    # Unified data section with use_dtn
    control_dict = {"data": {"mod1": {"type": "model", "use_dtn": True}}, "analysis": {}}

    with prefect_test_harness():
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
