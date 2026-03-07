# SPDX-License-Identifier: Apache-2.0
#
"""
Infrastructure abstraction for Dask clusters with native support for NCAR and NOAA platforms.
"""

import logging
import os

logger = logging.getLogger(__name__)

class ClusterFactory:
    """
    Factory class to create Dask clusters for different environments.
    Supports NCAR (Casper, Derecho) and NOAA RDHPCS (Hera, Jet, Orion, Hercules).
    """

    PLATFORMS = {
        'casper': {
            'cluster_type': 'slurm',
            'defaults': {
                'queue': 'casper',
                'project': os.getenv('PROJECT'),
                'cores': 1,
                'memory': '4GB',
                'walltime': '01:00:00'
            }
        },
        'derecho': {
            'cluster_type': 'pbs',
            'defaults': {
                'queue': 'main',
                'project': os.getenv('PROJECT'),
                'cores': 128,
                'memory': '256GB',
                'walltime': '01:00:00'
            }
        },
        'hera': {
            'cluster_type': 'slurm',
            'defaults': {
                'queue': 'batch',
                'cores': 40,
                'memory': '92GB',
                'walltime': '01:00:00'
            }
        },
        'jet': {
            'cluster_type': 'slurm',
            'defaults': {
                'queue': 'batch',
                'cores': 24,
                'memory': '60GB',
                'walltime': '01:00:00'
            }
        },
        'orion': {
            'cluster_type': 'slurm',
            'defaults': {
                'queue': 'batch',
                'cores': 40,
                'memory': '192GB',
                'walltime': '01:00:00'
            }
        },
        'hercules': {
            'cluster_type': 'slurm',
            'defaults': {
                'queue': 'batch',
                'cores': 80,
                'memory': '512GB',
                'walltime': '01:00:00'
            }
        }
    }

    @staticmethod
    def create_cluster(config):
        """
        Create and configure a Dask cluster based on the provided configuration.

        Parameters
        ----------
        config : dict
            Dask configuration dictionary.
            Expected keys:
                - cluster_type: 'local', 'slurm', 'pbs', 'lsf', 'fargate', etc.
                - platform: Optional, one of 'casper', 'derecho', 'hera', 'jet', 'orion', 'hercules'.
                - cluster_kwargs: dict of arguments passed to the cluster constructor.
                - adaptive: dict with 'enabled', 'minimum', 'maximum' for adaptive scaling.
                - scale: int, number of workers to scale to if adaptive is disabled.

        Returns
        -------
        cluster : dask.distributed.SpecCluster
            The initialized and configured Dask cluster.
        """
        platform = config.get('platform', '').lower()
        cluster_type = config.get('cluster_type', 'local').lower()
        cluster_kwargs = config.get('cluster_kwargs', {})
        adaptive_config = config.get('adaptive', {})
        scale = config.get('scale', None)

        # Apply platform-specific defaults
        if platform in ClusterFactory.PLATFORMS:
            logger.info(f"Applying defaults for platform: {platform}")
            platform_info = ClusterFactory.PLATFORMS[platform]
            cluster_type = platform_info['cluster_type']

            # Merge defaults with user-provided kwargs (user overrides defaults)
            defaults = platform_info['defaults'].copy()
            defaults.update(cluster_kwargs)
            cluster_kwargs = defaults

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
