import os
import warnings

class ClusterFactory:
    """
    Factory class to provision Dask clusters based on configuration.
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
            Additional keyword arguments for the cluster constructor.
        adaptive_kwargs : dict, optional
            Keyword arguments for the .adapt() method for dynamic scaling.

        Returns
        -------
        dask.distributed.SpecCluster
            The provisioned Dask cluster.
        """
        if cluster_kwargs is None:
            cluster_kwargs = {}
        if adaptive_kwargs is None:
            adaptive_kwargs = {}

        # Identify project/account codes from environment variables
        account = os.environ.get("ACCOUNT")
        project = os.environ.get("PROJECT")

        cluster = None

        if cluster_type.lower() == "local":
            from dask.distributed import LocalCluster
            cluster = LocalCluster(**cluster_kwargs)

        elif cluster_type.lower() == "slurm":
            from dask_jobqueue import SLURMCluster
            if account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = account
            cluster = SLURMCluster(**cluster_kwargs)

        elif cluster_type.lower() == "pbs":
            from dask_jobqueue import PBSCluster
            if account and "account" not in cluster_kwargs:
                cluster_kwargs["account"] = account
            cluster = PBSCluster(**cluster_kwargs)

        elif cluster_type.lower() == "lsf":
            from dask_jobqueue import LSFCluster
            if project and "project" not in cluster_kwargs:
                cluster_kwargs["project"] = project
            cluster = LSFCluster(**cluster_kwargs)

        elif cluster_type.lower() == "fargate":
            from dask_cloudprovider.aws import FargateCluster
            cluster = FargateCluster(**cluster_kwargs)

        elif cluster_type.lower() == "azure":
            from dask_cloudprovider.azure import AzureVMCluster
            cluster = AzureVMCluster(**cluster_kwargs)

        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        # Enable dynamic scaling if adaptive_kwargs are provided
        if adaptive_kwargs and hasattr(cluster, "adapt"):
            cluster.adapt(**adaptive_kwargs)

        return cluster
