# SPDX-License-Identifier: Apache-2.0
#
import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
import os
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

    def test_cluster_factory_platform_casper(self):
        config = {'platform': 'casper', 'scale': 2}
        with patch('dask_jobqueue.SLURMCluster') as mock_slurm:
            cluster = ClusterFactory.create_cluster(config)
            # Casper defaults: queue='casper', cores=1, memory='4GB', walltime='01:00:00'
            mock_slurm.assert_called_once_with(queue='casper', cores=1, memory='4GB', walltime='01:00:00')
            cluster.scale.assert_called_once_with(2)

    def test_cluster_factory_platform_derecho_override(self):
        config = {'platform': 'derecho', 'cluster_kwargs': {'walltime': '02:00:00'}, 'scale': 10}
        with patch('dask_jobqueue.PBSCluster') as mock_pbs:
            cluster = ClusterFactory.create_cluster(config)
            # Derecho defaults: queue='main', cores=128, memory='256GB', walltime='01:00:00'
            # Overridden: walltime='02:00:00'
            mock_pbs.assert_called_once_with(queue='main', cores=128, memory='256GB', walltime='02:00:00')
            cluster.scale.assert_called_once_with(10)

    def test_cluster_factory_platform_orion_alias(self):
        config = {'platform': 'msu', 'scale': 5}
        with patch('dask_jobqueue.SLURMCluster') as mock_slurm:
            cluster = ClusterFactory.create_cluster(config)
            # MSU is alias for Orion: queue='batch', cores=40, memory='192GB', walltime='01:00:00'
            mock_slurm.assert_called_once_with(queue='batch', cores=40, memory='192GB', walltime='01:00:00')
            cluster.scale.assert_called_once_with(5)

    def test_cluster_factory_platform_gaea_project_discovery(self):
        config = {'platform': 'gaea'}
        with patch.dict(os.environ, {'PROJECT': 'test_project'}):
            with patch('dask_jobqueue.SLURMCluster') as mock_slurm:
                ClusterFactory.create_cluster(config)
                # Gaea defaults: queue='batch', cores=32, memory='128GB', walltime='01:00:00'
                # Project discovery: account='test_project' (Slurm uses 'account')
                mock_slurm.assert_called_once_with(queue='batch', cores=32, memory='128GB', walltime='01:00:00', account='test_project')

    def test_cluster_factory_platform_ursa(self):
        config = {'platform': 'ursa', 'scale': 4}
        with patch('dask_jobqueue.SLURMCluster') as mock_slurm:
            cluster = ClusterFactory.create_cluster(config)
            # Ursa defaults: queue='batch', cores=44, memory='192GB', walltime='01:00:00'
            mock_slurm.assert_called_once_with(queue='batch', cores=44, memory='192GB', walltime='01:00:00')
            cluster.scale.assert_called_once_with(4)

    def test_cluster_factory_lsf_project(self):
        config = {'cluster_type': 'lsf'}
        with patch.dict(os.environ, {'PROJECT': 'test_project'}):
            with patch('dask_jobqueue.LSFCluster') as mock_lsf:
                ClusterFactory.create_cluster(config)
                # LSF uses 'project'
                mock_lsf.assert_called_once_with(project='test_project')

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
