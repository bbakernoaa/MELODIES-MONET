import os
import warnings

class ClusterFactory:
    """
    Factory class to provision Dask clusters based on configuration.
    Supports multi-partition and platform-specific configurations (NOAA RDHPCS, NCAR).
    """

    @staticmethod
    def get_cluster(cluster_type, cluster_kwargs=None, adaptive_kwargs=None):
        """
        Provision a Dask cluster.

        Parameters
        ----------
        cluster_type : str
            Type of cluster ('local', 'slurm', 'pbs', 'lsf', 'fargate', 'azure').
        cluster_kwargs : dict, optional
            Keyword arguments for the cluster constructor.
        adaptive_kwargs : dict, optional
            Keyword arguments for the .adapt() method.

        Returns
        -------
        dask.distributed.SpecCluster
            The provisioned Dask cluster.
        """
        if cluster_kwargs is None:
            cluster_kwargs = {}
        if adaptive_kwargs is None:
            adaptive_kwargs = {}

        # Identify project/account codes
        account = os.environ.get("ACCOUNT") or os.environ.get("PROJECT")

        cluster_type_lower = cluster_type.lower()

        if cluster_type_lower == "local":
            from dask.distributed import LocalCluster
            return LocalCluster(**cluster_kwargs)

        elif cluster_type_lower == "slurm":
            from dask_jobqueue import SLURMCluster
            if account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = account
            cluster = SLURMCluster(**cluster_kwargs)

        elif cluster_type_lower == "pbs":
            from dask_jobqueue import PBSCluster
            if account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = account
            cluster = PBSCluster(**cluster_kwargs)

        elif cluster_type_lower == "lsf":
            from dask_jobqueue import LSFCluster
            if account and "project" not in cluster_kwargs:
                cluster_kwargs["project"] = account
            cluster = LSFCluster(**cluster_kwargs)

        elif cluster_type_lower == "fargate":
            from dask_cloudprovider.aws import FargateCluster
            cluster = FargateCluster(**cluster_kwargs)

        elif cluster_type_lower == "azure":
            from dask_cloudprovider.azure import AzureVMCluster
            cluster = AzureVMCluster(**cluster_kwargs)

        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        # Enable dynamic scaling
        if adaptive_kwargs and hasattr(cluster, "adapt"):
            cluster.adapt(**adaptive_kwargs)

        return cluster

    @staticmethod
    def get_worker_spec(cluster_type, spec_kwargs):
        """
        Helper to generate worker specs for different partitions/QoS.
        """
        account = os.environ.get("ACCOUNT") or os.environ.get("PROJECT")

        if cluster_type.lower() == "slurm":
            if account and "account" not in spec_kwargs:
                spec_kwargs["account"] = account
        elif cluster_type.lower() == "lsf":
            if account and "project" not in spec_kwargs:
                spec_kwargs["project"] = account

        return spec_kwargs
