import networkx as nx

class GraphEngine:
    """
    GraphEngine class to manage the internal DAG for MELODIES-MONET.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_data_node(self, label, data_type):
        """
        Add a data node (model or observation) to the graph.

        Parameters
        ----------
        label : str
            The unique label for the data source.
        data_type : str
            'model' or 'obs'.
        """
        node_id = f"data_{label}"
        self.graph.add_node(node_id, type="data", data_type=data_type, label=label)
        return node_id

    def add_pairing_node(self, eval_label, model_label, obs_label):
        """
        Add a pairing node to the graph.

        Parameters
        ----------
        eval_label : str
            The unique label for the evaluation/pairing.
        model_label : str
            The label of the model data node.
        obs_label : str
            The label of the observation data node.
        """
        node_id = f"pair_{eval_label}"
        self.graph.add_node(node_id, type="pair", label=eval_label)

        # Dependencies: Pairing depends on Model and Obs data
        self.graph.add_edge(f"data_{model_label}", node_id)
        self.graph.add_edge(f"data_{obs_label}", node_id)
        return node_id

    def add_stats_node(self, group_label, pairing_labels):
        """
        Add a statistics node to the graph.

        Parameters
        ----------
        group_label : str
            The label for the stats group.
        pairing_labels : list of str
            List of evaluation/pairing labels this stats node depends on.
        """
        node_id = f"stats_{group_label}"
        self.graph.add_node(node_id, type="stats", label=group_label)

        for p_label in pairing_labels:
            self.graph.add_edge(f"pair_{p_label}", node_id)
        return node_id

    def add_plot_node(self, group_label, pairing_labels):
        """
        Add a plotting node to the graph.

        Parameters
        ----------
        group_label : str
            The label for the plotting group.
        pairing_labels : list of str
            List of evaluation/pairing labels this plot node depends on.
        """
        node_id = f"plot_{group_label}"
        self.graph.add_node(node_id, type="plot", label=group_label)

        for p_label in pairing_labels:
            self.graph.add_edge(f"pair_{p_label}", node_id)
        return node_id

    def validate_dag(self):
        """
        Check if the graph is a Directed Acyclic Graph (DAG).

        Returns
        -------
        bool
            True if it's a DAG, False otherwise.
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def get_execution_order(self):
        """
        Get a topological sort of the graph for execution.

        Returns
        -------
        list
            List of node IDs in execution order.
        """
        if not self.validate_dag():
            raise ValueError("Graph is not a DAG; cannot determine execution order.")
        return list(nx.topological_sort(self.graph))
