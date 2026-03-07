import networkx as nx
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

        # Support for data transfer/retrieval
        if self.ana.control_dict and "data_retrieval" in self.ana.control_dict:
            self.graph.add_node("transfer_data", func=self._transfer_data, resource_request="transfer")
            self.graph.add_edge("transfer_data", "open_models")
            self.graph.add_edge("transfer_data", "open_obs")

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

    def _transfer_data(self):
        """
        Implementation for data transfer/retrieval.
        Delegates to MONETIO's retrieval utilities or custom scripts.
        """
        print("Executing data transfer...")
        # Placeholder for actual data retrieval logic.
        # This would use the 'data_retrieval' section of the YAML.
        pass

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

    def run(self):
        """
        Execute the tasks in the DAG in topological order.
        """
        for node in nx.topological_sort(self.graph):
            print(f"Executing task: {node}")

            # Execution Check: Verify prerequisites
            if node == "pair_data":
                # Check if models have data loaded
                if not self.ana.models:
                    raise RuntimeError("Models must be opened before pairing.")
                for label, mod in self.ana.models.items():
                    if mod.obj is None:
                        raise RuntimeError(f"Model data for '{label}' has not been loaded.")

                # Check if observations have data loaded
                if not self.ana.obs:
                    raise RuntimeError("Observations must be opened before pairing.")
                for label, obs in self.ana.obs.items():
                    if obs.obj is None:
                        raise RuntimeError(f"Observation data for '{label}' has not been loaded.")

            if node == "stats" or node == "plotting":
                if not self.ana.paired:
                    # Check if 'read' is enabled, which might populate paired data
                    if self.ana.read and "paired" in self.ana.read:
                        print(f"Reading paired data from disk for {node}...")
                        self.ana.read_analysis()

                    if not self.ana.paired:
                        # Allow continuation if gridded pairing was executed
                        if "pair_gridded" not in self.graph.nodes:
                             raise RuntimeError(f"Paired data must be available before {node}.")

            func = self.graph.nodes[node]["func"]

            # Map YAML configuration keys directly to parameters
            # Each node can have specific parameters in the control dict
            node_params = self.ana.control_dict.get(node, {})
            if isinstance(node_params, list):
                # If the key points to a list (common in MM YAML for models/obs),
                # we don't treat it as kwargs for the function itself.
                node_params = {}

            if node == "pair_data":
                # Special handling for pair_data if it needs time_interval
                func()
            else:
                # For generic nodes, pass parameters from the control dict
                try:
                    import inspect
                    sig = inspect.signature(func)
                    if sig.parameters:
                        # Only pass params that the function expects
                        valid_params = {k: v for k, v in node_params.items() if k in sig.parameters}
                        func(**valid_params)
                    else:
                        func()
                except (ValueError, TypeError):
                    func()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        orchestrator = Orchestrator(sys.argv[1])
        orchestrator.run()
