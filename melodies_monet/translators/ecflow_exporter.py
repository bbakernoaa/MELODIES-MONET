import ecflow
import networkx as nx

class ecFlowExporter:
    """
    Translator class to convert MELODIES-MONET Orchestrator DAG to an ecFlow Suite.
    """

    def __init__(self, dag, suite_name="melodies_monet_suite", resource_mapper=None):
        self.dag = dag
        self.suite_name = suite_name
        self.suite = ecflow.Suite(self.suite_name)
        self.resource_mapper = resource_mapper

    def generate_suite(self):
        """
        Iterate through DAG nodes to create ecflow Tasks and map edges to triggers.
        """
        # Dictionary to keep track of ecflow Task objects for setting triggers
        tasks = {}

        # Create all tasks (nodes)
        for node in self.dag.nodes():
            task = self.suite.add_task(node)
            tasks[node] = task

            # Map node attributes to ecflow variables
            node_attrs = self.dag.nodes[node]
            resource_request = node_attrs.get("resource_request")

            if self.resource_mapper and resource_request:
                directives = self.resource_mapper.get_directives(resource_request)
                for key, value in directives.items():
                    task.add_variable(key.upper(), str(value))
            else:
                # Fallback to direct attributes
                if "memory" in node_attrs:
                    task.add_variable("MEMORY", node_attrs["memory"])
                if "cpu" in node_attrs:
                    task.add_variable("CPU", str(node_attrs["cpu"]))
                if "queue" in node_attrs:
                    task.add_variable("QUEUE", node_attrs["queue"])

        # Create all dependencies (edges) using task triggers
        for edge in self.dag.edges():
            parent_node, child_node = edge
            tasks[child_node].add_trigger(f"{parent_node} == complete")

        return self.suite

    def save_suite(self, filename):
        """
        Save the ecflow definition to a file.
        """
        defs = ecflow.Defs()
        defs.add_suite(self.generate_suite())
        defs.save_as_defs(filename)
        print(f"ecFlow suite saved to {filename}")

if __name__ == "__main__":
    # Example usage with a simple DAG
    G = nx.DiGraph()
    G.add_node("open_models", memory="16GB")
    G.add_node("open_obs", memory="16GB")
    G.add_node("pair_data", memory="64GB")
    G.add_edge("open_models", "pair_data")
    G.add_edge("open_obs", "pair_data")

    exporter = ecFlowExporter(G)
    # exporter.save_suite("mm_suite.def") # Requires ecflow library at runtime
