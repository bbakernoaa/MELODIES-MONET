from jinja2 import Template
import os

SNAKEMAKE_TEMPLATE = """
# Generated Snakefile for MELODIES-MONET workflow

rule all:
    input:
        {% for node in leaf_nodes %}
        "{{ node }}.complete"{% if not loop.last %},{% endif %}
        {% endfor %}

{% for node, dependencies in nodes_with_deps %}
rule {{ node }}:
    {% if dependencies %}
    input:
        {% for dep in dependencies %}
        "{{ dep }}.complete"{% if not loop.last %},{% endif %}
        {% endfor %}
    {% endif %}
    output:
        "{{ node }}.complete"
    {% if node_attrs[node] %}
    params:
        {% for key, value in node_attrs[node].items() %}
        {{ key }}="{{ value }}"{% if not loop.last %},{% endif %}
        {% endfor %}
    {% endif %}
    shell:
        "melodies-monet run-node {{ control_file }} {{ node }} && touch {{ node }}.complete"
{% endfor %}
"""

class SnakemakeExporter:
    """
    Exports a NetworkX DAG to a Snakemake Snakefile.
    """

    def __init__(self, graph, control_file="control.yaml"):
        """
        Initialize the exporter.
        """
        self.graph = graph
        self.control_file = control_file

    def generate_snakefile(self, output_file="Snakefile"):
        """
        Generate the Snakemake file from the DAG.
        """
        import networkx as nx

        # Leaf nodes (nodes with no out-edges)
        leaf_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]

        # Map dependencies for each node
        nodes_with_deps = []
        for node in nx.topological_sort(self.graph):
            dependencies = list(self.graph.predecessors(node))
            nodes_with_deps.append((node, dependencies))

        # Node attributes
        node_attrs = {}
        for node in self.graph.nodes():
            node_attrs[node] = {k: v for k, v in self.graph.nodes[node].items() if k != "func"}

        template = Template(SNAKEMAKE_TEMPLATE)
        rendered_snakefile = template.render(
            leaf_nodes=leaf_nodes,
            nodes_with_deps=nodes_with_deps,
            node_attrs=node_attrs,
            control_file=self.control_file
        )

        with open(output_file, "w") as f:
            f.write(rendered_snakefile)

        print(f"Snakemake Snakefile saved to {output_file}")
        return rendered_snakefile

if __name__ == "__main__":
    import networkx as nx
    # Quick test
    G = nx.DiGraph()
    G.add_node("open_models", memory="10GB")
    G.add_node("open_obs", memory="10GB")
    G.add_node("pair_data", memory="64GB")
    G.add_edge("open_models", "pair_data")
    G.add_edge("open_obs", "pair_data")

    exporter = SnakemakeExporter(G)
    exporter.generate_snakefile("Snakefile.test")
