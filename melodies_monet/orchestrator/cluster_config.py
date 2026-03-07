# SPDX-License-Identifier: Apache-2.0
#
"""
Infrastructure abstraction for Dask clusters.
"""

import logging

logger = logging.getLogger(__name__)

class ClusterFactory:
    """
    Factory class to create Dask clusters for different environments (HPC, Cloud, Local).
    """

    @staticmethod
    def create_cluster(config):
        """
        Create and configure a Dask cluster based on the provided configuration.

        Parameters
        ----------
        config : dict
            Dask configuration dictionary, usually from the 'analysis: dask' section of the control YAML.
            Expected keys:
                - cluster_type: 'local', 'slurm', 'pbs', 'lsf', 'fargate', etc.
                - cluster_kwargs: dict of arguments passed to the cluster constructor.
                - adaptive: dict with 'enabled', 'minimum', 'maximum' for adaptive scaling.
                - scale: int, number of workers to scale to if adaptive is disabled.

        Returns
        -------
        cluster : dask.distributed.SpecCluster
            The initialized and configured Dask cluster.
        """
        cluster_type = config.get('cluster_type', 'local').lower()
        cluster_kwargs = config.get('cluster_kwargs', {})
        adaptive_config = config.get('adaptive', {})
        scale = config.get('scale', None)

        logger.info(f"Initializing Dask cluster of type: {cluster_type}")

        if cluster_type == 'local':
            from dask.distributed import LocalCluster
            cluster = LocalCluster(**cluster_kwargs)
        elif cluster_type == 'slurm':
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster(**cluster_kwargs)
        elif cluster_type == 'pbs':
            from dask_jobqueue import PBSCluster
            cluster = PBSCluster(**cluster_kwargs)
        elif cluster_type == 'lsf':
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster(**cluster_kwargs)
        elif cluster_type == 'fargate':
            try:
                from dask_cloud_provider.aws import FargateCluster
            except ImportError:
                logger.error("dask-cloud-provider is required for FargateCluster")
                raise
            cluster = FargateCluster(**cluster_kwargs)
        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        # Configure scaling
        if adaptive_config.get('enabled', False):
            minimum = adaptive_config.get('minimum', 1)
            maximum = adaptive_config.get('maximum', 10)
            logger.info(f"Setting adaptive scaling: min={minimum}, max={maximum}")
            cluster.adapt(minimum=minimum, maximum=maximum)
        elif scale is not None:
            logger.info(f"Scaling cluster to {scale} workers")
            cluster.scale(scale)

        return cluster
