# SPDX-License-Identifier: Apache-2.0
#
import yaml
import os

def load_machines():
    """
    Load machine configurations from the YAML file.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "machines.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

MACHINES = load_machines()

def get_machine_config(machine_name):
    """
    Get the configuration for a specific machine.
    """
    config = MACHINES.get(machine_name.lower())
    if config and "alias" in config:
        return get_machine_config(config["alias"])
    return config

def get_scheduler_directives(machine_name, profile_name="standard", job_name="mm_task"):
    """
    Translate machine profile into scheduler directives.
    """
    config = get_machine_config(machine_name)
    if not config:
        return []

    profile = config["profiles"].get(profile_name, config["profiles"]["standard"])
    scheduler = config["scheduler"]

    directives = []
    if scheduler == "slurm":
        directives.append(f"#SBATCH --job-name={job_name}")
        for key, value in profile.items():
            if key == "partition":
                directives.append(f"#SBATCH --partition={value}")
            elif key == "time":
                directives.append(f"#SBATCH --time={value}")
            elif key == "mem":
                directives.append(f"#SBATCH --mem={value}")
            elif key == "nodes":
                directives.append(f"#SBATCH --nodes={value}")
            elif key == "ntasks":
                directives.append(f"#SBATCH --ntasks={value}")
            elif key == "account":
                directives.append(f"#SBATCH --account={value}")
    elif scheduler == "pbs":
        directives.append(f"#PBS -N {job_name}")
        for key, value in profile.items():
            if key == "q":
                directives.append(f"#PBS -q {value}")
            elif key == "walltime":
                directives.append(f"#PBS -l walltime={value}")
            elif key == "mem":
                directives.append(f"#PBS -l mem={value}")
            elif key == "nodes":
                # Basic PBS nodes/ncpus
                ncpus = profile.get("ncpus", 1)
                directives.append(f"#PBS -l select={value}:ncpus={ncpus}")
            elif key == "account":
                directives.append(f"#PBS -A {value}")

    return directives
