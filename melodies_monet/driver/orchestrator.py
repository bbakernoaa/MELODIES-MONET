# SPDX-License-Identifier: Apache-2.0
#
import networkx as nx
import pandas as pd
from melodies_monet.driver import analysis

class Orchestrator:
    """
    The Orchestrator class coordinates the execution of tasks in MELODIES-MONET.
    It uses NetworkX to build a Directed Acyclic Graph (DAG) of tasks.
    """

    def __init__(self, control_file):
        """
        Initialize the Orchestrator with a control file.
        """
        self.control_file = control_file
        self.ana = analysis.analysis()
        self.ana.read_control(control_file)
        self.graph = nx.DiGraph()
        self._build_dag()
        self._setup_dask()

    def _setup_dask(self):
        """
        Initialize Dask client if configured.
        Supports standard distributed Client and dask-jobqueue Clusters.
        """
        if self.ana.control_dict and "analysis" in self.ana.control_dict:
            dask_config = self.ana.control_dict["analysis"].get("dask", None)
            if not dask_config:
                return

            from dask.distributed import Client

            # Check if using dask-jobqueue
            if "jobqueue" in dask_config:
                jq_config = dask_config["jobqueue"]
                cluster_type = jq_config.get("type", "slurm").lower()
                cluster_kwargs = jq_config.get("kwargs", {})

                if cluster_type == "slurm":
                    from dask_jobqueue import SLURMCluster as Cluster
                elif cluster_type == "pbs":
                    from dask_jobqueue import PBSCluster as Cluster
                elif cluster_type == "lsf":
                    from dask_jobqueue import LSFCluster as Cluster
                else:
                    raise ValueError(f"Unsupported jobqueue type: {cluster_type}")

                self.dask_cluster = Cluster(**cluster_kwargs)

                # Handle scaling
                scale_config = jq_config.get("scale", {"jobs": 1})
                if "adaptive" in scale_config:
                    self.dask_cluster.adapt(**scale_config["adaptive"])
                else:
                    self.dask_cluster.scale(**scale_config)

                self.dask_client = Client(self.dask_cluster)
                print(f"Dask jobqueue cluster ({cluster_type}) initialized: {self.dask_client}")
            else:
                # Standard distributed Client (e.g., LocalCluster or existing scheduler)
                self.dask_client = Client(**dask_config)
                print(f"Dask client initialized: {self.dask_client}")

    def _build_dag(self):
        """
        Build the Directed Acyclic Graph (DAG) based on the control file.
        """
        # Define nodes
        self.graph.add_node("open_models", func=self.ana.open_models)
        self.graph.add_node("open_obs", func=self.ana.open_obs)
        self.graph.add_node("pair_data", func=self.ana.pair_data)
        self.graph.add_node("stats", func=self.ana.stats)
        self.graph.add_node("plotting", func=self.ana.plotting)

        # Define edges (dependencies)
        self.graph.add_edge("open_models", "pair_data")
        self.graph.add_edge("open_obs", "pair_data")
        self.graph.add_edge("pair_data", "stats")
        self.graph.add_edge("pair_data", "plotting")

        # Support for gridded-to-gridded pairing (e.g. models-to-models or sat-to-models)
        if self.ana.control_dict and "gridded_pairing" in self.ana.control_dict:
            self.graph.add_node("pair_gridded", func=self._pair_gridded)
            self.graph.add_edge("open_models", "pair_gridded")
            self.graph.add_edge("open_obs", "pair_gridded")
            self.graph.add_edge("pair_gridded", "stats")
            self.graph.add_edge("pair_gridded", "plotting")

    def _pair_gridded(self):
        """
        Implementation for gridded-to-gridded pairing.
        Supports model-to-model and gridded satellite-to-model comparisons.
        Uses MONET's regridding capabilities.
        """
        import monet as m

        print("Executing gridded pairing...")
        gridded_pairing_config = self.ana.control_dict.get("gridded_pairing", {})
        for pairing_name, config in gridded_pairing_config.items():
            data1_label = config.get("data1")
            data2_label = config.get("data2")
            data1_type = config.get("data1_type", "model")  # 'model' or 'obs'
            data2_type = config.get("data2_type", "model")  # 'model' or 'obs'
            variables = config.get("variables", "all")

            # Resolve data1
            if data1_type == "model":
                obj1 = self.ana.models.get(data1_label)
            else:
                obj1 = self.ana.obs.get(data1_label)

            # Resolve data2
            if data2_type == "model":
                obj2 = self.ana.models.get(data2_label)
            else:
                obj2 = self.ana.obs.get(data2_label)

            if obj1 is None or obj2 is None:
                print(f"Warning: Data source {data1_label} or {data2_label} not found.")
                continue

            print(f"Pairing {data1_label} and {data2_label}...")
            # Use MONET to combine gridded datasets.
            # This delegates to MONET's core pairing logic for gridded data (regidding/alignment).
            paired_obj = m.combine_gridded(obj1.obj, obj2.obj, variables=variables)

            from melodies_monet.driver.pair import pair

            p = pair()
            p.model = data1_label
            p.obs = data2_label
            p.obj = paired_obj
            self.ana.paired[pairing_name] = p

    def run(self, node=None):
        """
        Execute the tasks in the DAG.
        If node is specified, only that node (and its prerequisites if not in standalone) is run.
        """
        nodes_to_run = [node] if node else nx.topological_sort(self.graph)

        for n in nodes_to_run:
            print(f"Executing task: {n}")

            # Execution Check: Verify prerequisites and handle standalone loading
            if n == "pair_data":
                if not self.ana.models or not self.ana.obs:
                    print(f"Loading dependencies for {n}...")
                    self.ana.open_models(load_files=True)
                    self.ana.open_obs(load_files=True)

            if n == "stats" or n == "plotting":
                if not self.ana.paired:
                    print(f"Loading paired data for {n}...")
                    self.ana.read_analysis()

            func = self.graph.nodes[n]["func"]
            func()

            # Automatic state persistence for workflow handoff
            if node:  # If running in standalone (workflow mode)
                if n == "pair_data" or n == "pair_gridded":
                    print(f"Saving paired data results from {n}...")
                    self.ana.save_analysis()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        orchestrator = Orchestrator(sys.argv[1])
        orchestrator.run()
