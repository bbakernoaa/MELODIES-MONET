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

        # Support for gridded-to-gridded pairing (e.g. models-to-models)
        if self.ana.control_dict and "model_to_model" in self.ana.control_dict:
            self.graph.add_node("pair_models", func=self._pair_models)
            self.graph.add_edge("open_models", "pair_models")
            self.graph.add_edge("pair_models", "stats")
            self.graph.add_edge("pair_models", "plotting")

    def _pair_models(self):
        """
        Implementation for model-to-model pairing.
        Uses MONET's regridding capabilities.
        """
        import monet as m

        print("Executing model-to-model pairing...")
        model_to_model_config = self.ana.control_dict.get("model_to_model", {})
        for pairing_name, config in model_to_model_config.items():
            model1_label = config.get("model1")
            model2_label = config.get("model2")
            variables = config.get("variables", "all")

            if model1_label not in self.ana.models or model2_label not in self.ana.models:
                print(f"Warning: Models {model1_label} or {model2_label} not found.")
                continue

            mod1 = self.ana.models[model1_label]
            mod2 = self.ana.models[model2_label]

            print(f"Pairing {model1_label} and {model2_label}...")
            # Use MONET to combine gridded datasets.
            # This is a conceptual delegation to MONET's core pairing logic for gridded data.
            paired_obj = m.combine_gridded(mod1.obj, mod2.obj, variables=variables)

            from melodies_monet.driver.pair import pair

            p = pair()
            p.model = model1_label
            p.obs = model2_label  # Treating second model as 'obs' for the pair object
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
                        # Allow continuation if pair_models was executed
                        if "pair_models" not in self.graph.nodes:
                             raise RuntimeError(f"Paired data must be available before {node}.")

            func = self.graph.nodes[node]["func"]

            # Map YAML configuration keys directly to parameters if needed
            func()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        orchestrator = Orchestrator(sys.argv[1])
        orchestrator.run()
