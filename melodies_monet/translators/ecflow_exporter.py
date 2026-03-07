import os
try:
    import ecflow
except ImportError:
    ecflow = None

class EcflowExporter:
    """
    Exports a NetworkX DAG to an ecFlow Suite.
    """

    def __init__(self, graph, suite_name="melodies_monet"):
        """
        Initialize the exporter.
        """
        self.graph = graph
        self.suite_name = suite_name
        if ecflow is None:
            print("Warning: ecflow-python not found. Exporter will not function correctly.")

    def generate_suite(self, output_file=None):
        """
        Generate the ecFlow suite from the DAG.
        """
        if ecflow is None:
            raise RuntimeError("ecflow-python is required for ecFlow generation.")

        suite = ecflow.Suite(self.suite_name)

        # Map to store task objects for adding triggers
        tasks = {}

        # First pass: Create tasks
        for node in self.graph.nodes():
            task = suite.add_task(node)
            tasks[node] = task

            # Add node attributes as variables
            node_attrs = self.graph.nodes[node]
            for key, value in node_attrs.items():
                if key != "func":  # Don't add function object as a variable
                    task.add_variable(key.upper(), str(value))

        # Second pass: Add triggers (dependencies)
        for u, v in self.graph.edges():
            # v depends on u
            tasks[v].add_trigger(f"./{u} == complete")

        # Create suite definition
        defs = ecflow.Defs()
        defs.add_suite(suite)

        if output_file:
            # Check if directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            defs.save_as_defs(output_file)
            print(f"ecFlow suite saved to {output_file}")

        return defs

if __name__ == "__main__":
    import networkx as nx
    # Quick test
    G = nx.DiGraph()
    G.add_node("open_models", memory="10GB")
    G.add_node("open_obs", memory="10GB")
    G.add_node("pair_data", memory="64GB")
    G.add_edge("open_models", "pair_data")
    G.add_edge("open_obs", "pair_data")

    exporter = EcflowExporter(G)
    try:
        exporter.generate_suite("test.def")
    except Exception as e:
        print(f"Error generating suite: {e}")
