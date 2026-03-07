from jinja2 import Template
import networkx as nx

class SnakemakeExporter:
    """
    Translator class to generate a Snakemake file from the Orchestrator DAG.
    """

    def __init__(self, dag, template_str=None, resource_mapper=None, scheduler="slurm"):
        self.dag = dag
        self.template_str = template_str if template_str else self._get_default_template()
        self.resource_mapper = resource_mapper
        self.scheduler = scheduler

    def _get_default_template(self):
        """
        Default Jinja2 template for the Snakefile.
        """
        return """
# MM Data Contract - Snakemake file
# Each rule represents a node in the Orchestrator DAG.

rule all:
    input:
        {% for target in targets -%}
        "{{ target }}",
        {% endfor %}

{% for node, attr in nodes.items() -%}
rule {{ node }}:
    input:
        {% for parent in attr.parents -%}
        "{{ parent }}.done",
        {% endfor %}
    output:
        "{{ node }}.done"
    {% if attr.resources -%}
    resources:
        {% for key, value in attr.resources.items() -%}
        {{ key }}="{{ value }}",
        {% endfor %}
    {%- endif %}
    {% if attr.directives -%}
    # Scheduler directives for this rule:
    {% for directive in attr.directives -%}
    # {{ directive }}
    {% endfor %}
    {%- endif %}
    shell:
        "melodies-monet run-node {{ node }} --config control.yaml && touch {{ node }}.done"
{% endfor %}
"""

    def generate_snakefile(self):
        """
        Populate the Jinja2 template with the DAG information.
        """
        template = Template(self.template_str)
        nodes = {}
        targets = []

        # Determine targets (nodes with no out-edges)
        for node in self.dag.nodes():
            if self.dag.out_degree(node) == 0:
                targets.append(f"{node}.done")

            # Identify parents of each node for inputs
            parents = list(self.dag.predecessors(node))

            # Get resource mapping for the node
            node_attrs = self.dag.nodes[node]
            resource_request = node_attrs.get("resource_request")
            resources = {}
            directives = []
            if self.resource_mapper and resource_request:
                resources = self.resource_mapper.get_directives(resource_request)
                directives = self.resource_mapper.translate_to_hpc(resource_request, scheduler=self.scheduler)

            nodes[node] = {"parents": parents, "resources": resources, "directives": directives}

        return template.render(nodes=nodes, targets=targets)

    def save_snakefile(self, filename="Snakefile"):
        """
        Save the generated Snakemake content to a file.
        """
        content = self.generate_snakefile()
        with open(filename, "w") as f:
            f.write(content)
        print(f"Snakemake file saved to {filename}")

if __name__ == "__main__":
    # Example usage with a simple DAG
    G = nx.DiGraph()
    G.add_edge("open_models", "pair_data")
    G.add_edge("open_obs", "pair_data")
    G.add_edge("pair_data", "stats")
    G.add_edge("pair_data", "plotting")

    exporter = SnakemakeExporter(G)
    print(exporter.generate_snakefile())
