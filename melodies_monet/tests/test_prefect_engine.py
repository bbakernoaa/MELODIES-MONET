# SPDX-License-Identifier: Apache-2.0
#
import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
from melodies_monet.orchestrator.cluster_config import ClusterFactory
from melodies_monet.orchestrator.prefect_engine import mm_prefect_flow

class TestPrefectEngine(unittest.TestCase):

    def test_cluster_factory_local(self):
        config = {'cluster_type': 'local', 'cluster_kwargs': {'n_workers': 2}}
        with patch('dask.distributed.LocalCluster') as mock_local:
            ClusterFactory.create_cluster(config)
            mock_local.assert_called_once_with(n_workers=2)

    def test_cluster_factory_slurm(self):
        config = {'cluster_type': 'slurm', 'cluster_kwargs': {'queue': 'regular'}, 'scale': 5}
        with patch('dask_jobqueue.SLURMCluster') as mock_slurm:
            cluster = ClusterFactory.create_cluster(config)
            mock_slurm.assert_called_once_with(queue='regular')
            cluster.scale.assert_called_once_with(5)

    @patch('melodies_monet.orchestrator.prefect_engine.run_mm_node.submit')
    @patch('melodies_monet.orchestrator.prefect_engine.Client')
    @patch('melodies_monet.orchestrator.prefect_engine.ClusterFactory.create_cluster')
    def test_flow_construction(self, mock_create_cluster, mock_client, mock_submit):
        # Mock Orchestrator
        orchestrator = MagicMock()
        orchestrator.ana.control_dict = {
            'analysis': {
                'dask': {'cluster_type': 'local'}
            }
        }

        # Build a simple mock DAG
        orchestrator.graph = nx.DiGraph()
        orchestrator.graph.add_node("task1", func=lambda: print("task1"))
        orchestrator.graph.add_node("task2", func=lambda: print("task2"))
        orchestrator.graph.add_edge("task1", "task2")

        # Run flow construction
        mm_prefect_flow(orchestrator)

        # Check if tasks were submitted in order
        self.assertEqual(mock_submit.call_count, 2)

        # Verify first task submission (no dependencies)
        args, kwargs = mock_submit.call_args_list[0]
        self.assertEqual(kwargs['node_name'], 'task1')
        self.assertEqual(kwargs['wait_for'], [])

        # Verify second task submission (depends on first)
        args, kwargs = mock_submit.call_args_list[1]
        self.assertEqual(kwargs['node_name'], 'task2')
        self.assertEqual(len(kwargs['wait_for']), 1)

if __name__ == '__main__':
    unittest.main()
