
import networkx as nx
from melodies_monet.translators.snakemake_exporter import SnakemakeExporter
from melodies_monet.translators.resources import HPCResourceMapper
import yaml
import os

# Create dummy resource YAML
resources = {
    "high-mem": {
        "memory": "128G",
        "cpu": 8
    },
    "standard": {
        "memory": "16G",
        "cpu": 2
    }
}
with open("hpc_resources.yaml", "w") as f:
    yaml.dump(resources, f)

# Create DAG
G = nx.DiGraph()
G.add_node("transfer_data", resource_request="transfer")
G.add_node("open_models", resource_request="standard")
G.add_node("pair_data", resource_request="high-mem")
G.add_edge("transfer_data", "open_models")
G.add_edge("open_models", "pair_data")

# Export
mapper = HPCResourceMapper(config_file="hpc_resources.yaml")
exporter = SnakemakeExporter(G, resource_mapper=mapper)
snakefile_content = exporter.generate_snakefile()

print("Generated Snakefile:")
print(snakefile_content)

# Verification
assert 'resources:' in snakefile_content
assert 'memory="128G"' in snakefile_content
assert 'cpu="8"' in snakefile_content
assert 'rule transfer_data:' in snakefile_content

print("Verification successful!")

# Cleanup
os.remove("hpc_resources.yaml")
