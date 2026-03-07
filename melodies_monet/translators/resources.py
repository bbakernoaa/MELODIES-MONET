import yaml
import os

class HPCResourceMapper:
    """
    Manages mapping of abstract MM resource requests to specific HPC directives.
    Loads configurations from YAML files.
    """

    def __init__(self, hpc_type=None, config_file=None):
        self.hpc_type = hpc_type
        self.mapping = {}
        if config_file:
            self.load_config(config_file)
        elif hpc_type:
            # Look for default configs if available
            pass

    def load_config(self, config_file):
        """
        Load resource mapping from a YAML file.
        """
        with open(config_file, 'r') as f:
            self.mapping = yaml.safe_load(f)

    def get_directives(self, resource_request):
        """
        Get HPC-specific directives for a given abstract resource request (e.g., 'high-mem').
        """
        return self.mapping.get(resource_request, {})

    def translate_to_hpc(self, resource_request, scheduler="slurm"):
        """
        Translate a request into scheduler-specific directives.
        Supported schedulers: slurm, pbs, lsf.
        """
        res = self.get_directives(resource_request)
        directives = []

        if scheduler == "slurm":
            if "memory" in res:
                directives.append(f"#SBATCH --mem={res['memory']}")
            if "cpu" in res:
                directives.append(f"#SBATCH --cpus-per-task={res['cpu']}")
            if "walltime" in res:
                directives.append(f"#SBATCH --time={res['walltime']}")
            if "queue" in res:
                directives.append(f"#SBATCH --partition={res['queue']}")
            if "project" in res:
                directives.append(f"#SBATCH --account={res['project']}")

        elif scheduler == "pbs":
            if "memory" in res:
                directives.append(f"#PBS -l mem={res['memory']}")
            if "cpu" in res:
                directives.append(f"#PBS -l ncpus={res['cpu']}")
            if "walltime" in res:
                directives.append(f"#PBS -l walltime={res['walltime']}")
            if "queue" in res:
                directives.append(f"#PBS -q {res['queue']}")
            if "project" in res:
                directives.append(f"#PBS -A {res['project']}")

        elif scheduler == "lsf":
            if "memory" in res:
                directives.append(f"#BSUB -M {res['memory']}")
            if "cpu" in res:
                directives.append(f"#BSUB -n {res['cpu']}")
            if "walltime" in res:
                directives.append(f"#BSUB -W {res['walltime']}")
            if "queue" in res:
                directives.append(f"#BSUB -q {res['queue']}")
            if "project" in res:
                directives.append(f"#BSUB -P {res['project']}")

        return directives

# Example YAML structure for an HPC suite:
# high-mem:
#   memory: 128G
#   cpu: 8
#   walltime: "02:00:00"
#   queue: bigmem
# standard:
#   memory: 16G
#   cpu: 2
#   walltime: "01:00:00"
#   queue: normal
