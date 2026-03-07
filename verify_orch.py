
import yaml
import os
from melodies_monet.driver.orchestrator import Orchestrator

# Create a dummy control file
control = {
    "analysis": {
        "start_time": "2019-09-01 00:00:00",
        "end_time": "2019-09-01 23:00:00",
        "output_dir": "./output",
        "debug": True
    },
    "model": {
        "model1": {
            "mod_type": "cmaq",
            "files": "dummy_model1.nc",
            "mapping": {"obs1": {"O3": "O3"}},
            "variables": {"O3": {}}
        },
        "model2": {
            "mod_type": "cmaq",
            "files": "dummy_model2.nc",
            "mapping": {},
            "variables": {"O3": {}}
        }
    },
    "obs": {
        "obs1": {
            "obs_type": "pt_sfc",
            "filename": "dummy_obs1.nc",
            "variables": {"O3": {}}
        }
    },
    "gridded_pairing": {
        "m1_m2_pair": {
            "data1": "model1",
            "data2": "model2",
            "data1_type": "model",
            "data2_type": "model",
            "variables": ["O3"]
        }
    }
}

with open("test_control.yaml", "w") as f:
    yaml.dump(control, f)

try:
    orch = Orchestrator("test_control.yaml")
    print("Nodes in DAG:", orch.graph.nodes)
    print("Edges in DAG:", orch.graph.edges)

    # Check if pair_gridded is in the graph
    assert "pair_gridded" in orch.graph.nodes
    assert ("open_models", "pair_gridded") in orch.graph.edges

    print("Verification successful: DAG correctly constructed with gridded pairing.")
finally:
    if os.path.exists("test_control.yaml"):
        os.remove("test_control.yaml")
