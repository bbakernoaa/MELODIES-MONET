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

        # Support for gridded-to-gridded pairing (legacy gridded_pairing or new pairings)
        has_gridded = self.ana.control_dict and "gridded_pairing" in self.ana.control_dict
        if not has_gridded and self.ana.control_dict and "pairings" in self.ana.control_dict:
            for pair_label, config in self.ana.control_dict["pairings"].items():
                if config.get("data1_type") == "model" and config.get("data2_type") == "model":
                    has_gridded = True
                    break

        if has_gridded:
            # We don't need a separate node anymore as pair_data handles it,
            # but we keep the logic here for backward compatibility or explicit gridded step if needed.
            # Actually, it's better to unify it in pair_data.
            pass

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
                    print("Warning: No models defined.")
                for label, mod in self.ana.models.items():
                    if mod.obj is None:
                        raise RuntimeError(f"Model data for '{label}' has not been loaded.")

                # Check if observations have data loaded
                if not self.ana.obs:
                    print("Warning: No observations defined.")
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
                        raise RuntimeError(f"Paired data must be available before {node}.")

            func = self.graph.nodes[node]["func"]

            # Map YAML configuration keys directly to parameters if needed
            func()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        orchestrator = Orchestrator(sys.argv[1])
        orchestrator.run()
