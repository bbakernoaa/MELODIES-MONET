import os
import socket

PLATFORM_CONFIGS = {
    "hera": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 40, "memory": "128GB"},
        "worker_groups": {
            "compute": {"queue": "batch", "cores": 40, "memory": "128GB"},
            "dtn": {
                "queue": "service",
                "cores": 1,
                "memory": "8GB",
                "worker_extra_args": ["--resources dtn=1"],
            },
        },
    },
    "jet": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 24, "memory": "64GB"},
        "worker_groups": {
            "compute": {"queue": "batch", "cores": 24, "memory": "64GB"},
            "dtn": {
                "queue": "service",
                "cores": 1,
                "memory": "8GB",
                "worker_extra_args": ["--resources dtn=1"],
            },
        },
    },
    "orion": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "orion", "cores": 40, "memory": "192GB"},
        "worker_groups": {"compute": {"queue": "orion", "cores": 40, "memory": "192GB"}},
    },
    "hercules": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "hercules", "cores": 80, "memory": "256GB"},
        "worker_groups": {"compute": {"queue": "hercules", "cores": 80, "memory": "256GB"}},
    },
    "gaea-c5": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 32, "memory": "128GB"},
        "worker_groups": {"compute": {"queue": "batch", "cores": 32, "memory": "128GB"}},
    },
    "gaea-c6": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 64, "memory": "256GB"},
        "worker_groups": {"compute": {"queue": "batch", "cores": 64, "memory": "256GB"}},
    },
    "gaea": {  # Default to c5 for backward compatibility
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 32, "memory": "128GB"},
        "worker_groups": {"compute": {"queue": "batch", "cores": 32, "memory": "128GB"}},
    },
    "ursa": {
        "cluster_type": "slurm",
        "cluster_kwargs": {"queue": "batch", "cores": 32, "memory": "128GB"},
        "worker_groups": {"compute": {"queue": "batch", "cores": 32, "memory": "128GB"}},
    },
    "casper": {
        "cluster_type": "pbs",
        "cluster_kwargs": {
            "queue": "casper",
            "resource_spec": "select=1:ncpus=4:mem=16GB",
        },
        "worker_groups": {"compute": {"queue": "casper", "resource_spec": "select=1:ncpus=4:mem=16GB"}},
    },
    "derecho": {
        "cluster_type": "pbs",
        "cluster_kwargs": {
            "queue": "main",
            "resource_spec": "select=1:ncpus=128:mem=256GB",
        },
        "worker_groups": {
            "compute": {
                "queue": "main",
                "resource_spec": "select=1:ncpus=128:mem=256GB",
            }
        },
    },
}


class ClusterFactory:
    """
    Factory class to provision Dask clusters based on configuration.
    """

    @staticmethod
    def get_cluster(
        cluster_type=None,
        cluster_kwargs=None,
        adaptive_kwargs=None,
        platform=None,
        account=None,
        project=None,
    ):
        """
        Get a Dask cluster instance based on the configuration.

        Parameters
        ----------
        cluster_type : str, optional
            The type of cluster to create (e.g., 'local', 'slurm', 'pbs', 'lsf', 'fargate', 'azure').
        cluster_kwargs : dict, optional
            Keyword arguments to pass to the cluster constructor.
        adaptive_kwargs : dict, optional
            Keyword arguments for the cluster's adapt() method.
        platform : str, optional
            A predefined platform name (e.g., 'hera', 'jet', 'orion') to use default settings.
        account : str, optional
            HPC account or project identifier.
        project : str, optional
            HPC project identifier (alias for account).

        Returns
        -------
        dask.distributed.SpecCluster or dask.distributed.LocalCluster or dask_jobqueue.Cluster
            The provisioned Dask cluster instance.
        """
        if cluster_kwargs is None:
            cluster_kwargs = {}
        if adaptive_kwargs is None:
            adaptive_kwargs = {}

        # Platform Detection
        if platform is None:
            hostname = socket.gethostname()
            # Sort by length descending to catch gaea-c5 before gaea
            sorted_platforms = sorted(PLATFORM_CONFIGS.keys(), key=len, reverse=True)
            for p_key in sorted_platforms:
                if p_key in hostname:
                    platform = p_key
                    break

        worker_groups = {}
        if platform and platform in PLATFORM_CONFIGS:
            p_config = PLATFORM_CONFIGS[platform]
            if cluster_type is None:
                cluster_type = p_config["cluster_type"]
            for k, v in p_config.get("cluster_kwargs", {}).items():
                if k not in cluster_kwargs:
                    cluster_kwargs[k] = v
            worker_groups = p_config.get("worker_groups", {}).copy()

        if cluster_type is None:
            cluster_type = "local"
        cluster_type_lower = cluster_type.lower()
        final_account = account or os.environ.get("ACCOUNT") or project or os.environ.get("PROJECT")

        if worker_groups:
            from dask.distributed import SpecCluster

            job_cls = None
            if cluster_type_lower == "slurm":
                from dask_jobqueue.slurm import SLURMJob

                job_cls = SLURMJob
            elif cluster_type_lower == "pbs":
                from dask_jobqueue.pbs import PBSJob

                job_cls = PBSJob
            elif cluster_type_lower == "lsf":
                from dask_jobqueue.lsf import LSFJob

                job_cls = LSFJob

            if job_cls:
                worker_specs = {}
                for name, spec in worker_groups.items():
                    spec = spec.copy()
                    if final_account:
                        if cluster_type_lower == "lsf":
                            spec.setdefault("project", final_account)
                        else:
                            spec.setdefault("account", final_account)
                    worker_specs[name] = {"cls": job_cls, "options": spec}
                return SpecCluster(worker_specs)

        if cluster_type_lower == "local":
            from dask.distributed import LocalCluster

            return LocalCluster(**cluster_kwargs)

        elif cluster_type_lower == "slurm":
            from dask_jobqueue import SLURMCluster

            if final_account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = final_account
            cluster = SLURMCluster(**cluster_kwargs)

        elif cluster_type_lower == "pbs":
            from dask_jobqueue import PBSCluster

            if final_account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = final_account
            cluster = PBSCluster(**cluster_kwargs)

        elif cluster_type_lower == "lsf":
            from dask_jobqueue import LSFCluster

            if final_account and "project" not in cluster_kwargs:
                cluster_kwargs["project"] = final_account
            cluster = LSFCluster(**cluster_kwargs)

        elif cluster_type_lower == "fargate":
            from dask_cloudprovider.aws import FargateCluster

            cluster = FargateCluster(**cluster_kwargs)

        elif cluster_type_lower == "azure":
            from dask_cloudprovider.azure import AzureVMCluster

            cluster = AzureVMCluster(**cluster_kwargs)

        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        if adaptive_kwargs and hasattr(cluster, "adapt"):
            cluster.adapt(**adaptive_kwargs)

        return cluster
