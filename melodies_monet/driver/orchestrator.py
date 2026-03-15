# SPDX-License-Identifier: Apache-2.0
#
import os

import monet as m
import networkx as nx
import pandas as pd

from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair


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


class orchestrator:
    """The orchestrator class. High-level orchestrator for evaluations."""

    def __init__(self):
        self.control = "control.yaml"
        self.control_dict = None
        self.graph_engine = GraphEngine()
        self.data = {}
        self.paired = {}
        self.start_time = None
        self.end_time = None
        self.time_intervals = None
        self.download_maps = True
        self.output_dir = None
        self.output_dir_save = None
        self.output_dir_read = None
        self.debug = False
        self.save = None
        self.read = None
        self.regrid = False
        self.target_grid = None
        self.add_logo = True
        self.pairing_kwargs = {}
        self.platform = None
        self.account = None
        self.project = None

    def read_control(self, control=None):
        """
        Read and parse the control YAML file.

        Parameters
        ----------
        control : str, optional
            Path to the control YAML file. If not provided, uses the default 'control.yaml'.
        """
        import yaml

        if control is not None:
            self.control = control
        with open(self.control, "r") as stream:
            self.control_dict = yaml.safe_load(stream)

        self._migrate_control_dict()
        self._build_graph()
        self.graph_engine.validate_dag()

        # Basic settings
        self.start_time = pd.Timestamp(self.control_dict["analysis"]["start_time"])
        self.end_time = pd.Timestamp(self.control_dict["analysis"]["end_time"])
        self.output_dir = os.path.expandvars(self.control_dict["analysis"]["output_dir"])
        self.output_dir_save = os.path.expandvars(self.control_dict["analysis"].get("output_dir_save", self.output_dir))
        self.output_dir_read = os.path.expandvars(self.control_dict["analysis"].get("output_dir_read", self.output_dir))
        self.debug = self.control_dict["analysis"].get("debug", False)
        self.save = self.control_dict["analysis"].get("save")
        self.read = self.control_dict["analysis"].get("read")
        self.add_logo = self.control_dict["analysis"].get("add_logo", True)
        self.regrid = self.control_dict["analysis"].get("regrid", False)
        self.target_grid = self.control_dict["analysis"].get("target_grid")
        self.pairing_kwargs = self.control_dict["analysis"].get("pairing_kwargs", {})
        self.platform = self.control_dict["analysis"].get("platform")
        self.account = self.control_dict["analysis"].get("account")
        self.project = self.control_dict["analysis"].get("project")

        # Time intervals for chunking
        if "time_interval" in self.control_dict["analysis"]:
            # Ensure freq is lowercase for Pandas 3.0+ compatibility
            freq = self.control_dict["analysis"]["time_interval"]
            if isinstance(freq, str):
                freq = freq.lower()

            time_stamps = pd.date_range(
                start=self.start_time,
                end=self.end_time,
                freq=freq,
            )
            if time_stamps[-1] < pd.Timestamp(self.end_time):
                time_stamps = time_stamps.append(pd.DatetimeIndex([self.end_time]))
            self.time_intervals = [[time_stamps[n], time_stamps[n + 1]] for n in range(len(time_stamps) - 1)]

    def open_data(self, time_interval=None, load_files=True):
        """
        Open and load data sources defined in the control file.

        Parameters
        ----------
        time_interval : list of pd.Timestamp, optional
            A list containing [start_time, end_time] to subset the data.
        load_files : bool, optional
            Whether to actually load the data from files. Default is True.
        """
        if "data" in self.control_dict:
            for label, cfg in self.control_dict["data"].items():
                inst = Data(label=label)
                inst.from_dict(cfg)

                if load_files:
                    inst.load(time_interval=time_interval)
                self.data[label] = inst

    def pair_data(self):
        """
        Perform data pairing for all evaluations defined in the control file.
        Leverages monet.pair for the core pairing logic.
        """
        if "evaluations" in self.control_dict:
            for eval_label, cfg in self.control_dict["evaluations"].items():
                mod_label = cfg.get("model", cfg.get("exp"))
                ref_label = cfg.get("obs", cfg.get("ref"))
                mod = self.data[mod_label]
                ref = self.data[ref_label]

                # Mapping and subsetting
                mapping = cfg.get("mapping")
                if mapping is None:
                    mapping = mod.mapping.get(ref_label, {}) if mod.mapping else {}
                keys = list(mapping.keys())
                ref_vars = list(mapping.values())
                mod_vars = list(mod.variable_dict.keys()) if mod.variable_dict else []

                # Model variables for this pairing
                model_obj = mod.obj[list(set(keys + mod_vars))]

                # Perform pairing using the monet accessor
                # We use combine_point if available, otherwise fallback to monet.pair
                if hasattr(model_obj.monet, "combine_point"):
                    paired_data = model_obj.monet.combine_point(
                        ref.obj,
                        suffix=mod.label,
                        type=ref.obs_type.lower(),
                        **self.pairing_kwargs.get(ref.obs_type.lower(), {}),
                    )
                else:
                    # Fallback to monet.pair if combine_point is not available
                    # We check if monet has pair attribute
                    pair_func = getattr(m, "pair", None)
                    if pair_func is None:
                        # try to import it
                        from monet import pair as pair_func

                    paired_data = pair_func(
                        model_obj,
                        ref.obj,
                        suffix=mod.label,
                        type=ref.obs_type.lower(),
                        **self.pairing_kwargs.get(ref.obs_type.lower(), {}),
                    )

                p_inst = pair()
                p_inst.ref = ref.label
                p_inst.model = mod.label
                p_inst.model_vars = keys
                p_inst.ref_vars = ref_vars
                p_inst.type = ref.obs_type.lower()
                p_inst.obj = paired_data
                p_inst.model_plot_kwargs = mod.plot_kwargs
                p_inst.ref_plot_kwargs = ref.plot_kwargs

                self.paired[eval_label] = p_inst

    def _build_graph(self):
        """
        Build the internal Directed Acyclic Graph (DAG) using GraphEngine.
        """
        if "data" in self.control_dict:
            for label, cfg in self.control_dict["data"].items():
                self.graph_engine.add_data_node(label, cfg.get("type", "model"))

        if "evaluations" in self.control_dict:
            for eval_label, cfg in self.control_dict["evaluations"].items():
                mod_label = cfg.get("model", cfg.get("exp"))
                obs_label = cfg.get("obs", cfg.get("ref"))
                self.graph_engine.add_pairing_node(eval_label, mod_label, obs_label)

        if "stats" in self.control_dict:
            for group_label, cfg in self.control_dict["stats"].items():
                pairing_labels = cfg.get("data", [])
                self.graph_engine.add_stats_node(group_label, pairing_labels)

        if "plotting" in self.control_dict:
            for group_label, cfg in self.control_dict["plotting"].items():
                pairing_labels = cfg.get("data", [])
                self.graph_engine.add_plot_node(group_label, pairing_labels)

        if not self.graph_engine.validate_dag():
            raise ValueError("The constructed graph is not a Directed Acyclic Graph (DAG).")

    def _migrate_control_dict(self):
        """
        Transparently upgrade configurations to the unified data schema.
        Consolidates 'model', 'models', and 'obs' into 'data' and renames 'plots' to 'plotting'.
        """
        if "model" in self.control_dict:
            m_cfg = self.control_dict.pop("model")
            for k, v in m_cfg.items():
                v["type"] = "model"
                self.control_dict.setdefault("data", {})[k] = v

        if "models" in self.control_dict:
            m_cfg = self.control_dict.pop("models")
            for k, v in m_cfg.items():
                v["type"] = "model"
                self.control_dict.setdefault("data", {})[k] = v

        if "obs" in self.control_dict:
            o_cfg = self.control_dict.pop("obs")
            for k, v in o_cfg.items():
                v["type"] = "obs"
                self.control_dict.setdefault("data", {})[k] = v

        if "plots" in self.control_dict:
            self.control_dict.setdefault("plotting", {}).update(self.control_dict.pop("plots"))

        if "evaluations" not in self.control_dict and "data" in self.control_dict:
            self.control_dict["evaluations"] = {}
            for label, cfg in self.control_dict["data"].items():
                if "mapping" in cfg:
                    for obs_label, mapping in cfg["mapping"].items():
                        eval_label = f"{obs_label}_{label}"
                        self.control_dict["evaluations"][eval_label] = {
                            "exp": label,
                            "ref": obs_label,
                            "mapping": mapping,
                        }

        # Expand evaluations if exp or ref are lists
        if "evaluations" in self.control_dict:
            new_evals = {}
            expanded_map = {}  # old_eval_label -> [new_eval_labels]
            for eval_label, cfg in self.control_dict["evaluations"].items():
                exps = cfg.get("exp", cfg.get("model"))
                refs = cfg.get("ref", cfg.get("obs"))

                if isinstance(exps, list) or isinstance(refs, list):
                    if not isinstance(exps, list):
                        exps = [exps]
                    if not isinstance(refs, list):
                        refs = [refs]

                    expanded_labels = []
                    for e in exps:
                        for r in refs:
                            new_label = f"{r}_{e}"
                            expanded_labels.append(new_label)
                            new_evals[new_label] = cfg.copy()
                            if "exp" in cfg:
                                new_evals[new_label]["exp"] = e
                            if "model" in cfg:
                                new_evals[new_label]["model"] = e
                            if "ref" in cfg:
                                new_evals[new_label]["ref"] = r
                            if "obs" in cfg:
                                new_evals[new_label]["obs"] = r
                    expanded_map[eval_label] = expanded_labels
                else:
                    new_evals[eval_label] = cfg

            self.control_dict["evaluations"] = new_evals

            # Update data lists in plotting and stats
            for task in ["plotting", "stats"]:
                if task in self.control_dict:
                    for grp in self.control_dict[task].values():
                        if "data" in grp:
                            new_data = []
                            for d in grp["data"]:
                                if d in expanded_map:
                                    new_data.extend(expanded_map[d])
                                else:
                                    new_data.append(d)
                            grp["data"] = new_data

    def save_analysis(self):
        """
        Save analysis results (e.g., paired data) to disk as configured.
        """
        if self.save:
            for attr, cfg in self.save.items():
                obj = getattr(self, attr)
                if cfg["method"] == "netcdf":
                    from melodies_monet.util.write_util import write_analysis_ncf

                    write_analysis_ncf(
                        obj=obj,
                        output_dir=self.output_dir_save,
                        fn_prefix=cfg.get("prefix", ""),
                        keep_groups=cfg.get("data", "all"),
                    )
                elif cfg["method"] == "pkl":
                    from melodies_monet.util.write_util import write_pkl

                    write_pkl(
                        obj=obj,
                        output_name=os.path.join(self.output_dir_save, cfg["output_name"]),
                    )

    def plotting(self):
        """
        Execute plotting tasks as defined in the control file.
        Delegates to monet-plots.
        """
        if "plotting" in self.control_dict:
            from melodies_monet.util.bridge import create_plots

            for group_label, cfg in self.control_dict["plotting"].items():
                # Propagate global analysis settings
                cfg_with_global = {
                    "output_dir": self.output_dir,
                    "debug": self.debug,
                    "add_logo": self.add_logo,
                    **cfg,
                }
                # Create a copy of needed pairs to avoid modifying the original objects if needed,
                # but here we just need to pass them to monet_plots.
                needed_pairs = {k: self.paired[k] for k in cfg.get("data", []) if k in self.paired}
                create_plots(needed_pairs, **cfg_with_global)

    def stats(self):
        """
        Execute statistics tasks as defined in the control file.
        Delegates to monet-stats.
        """
        if "stats" in self.control_dict:
            from melodies_monet.util.bridge import compute_stats

            for group_label, cfg in self.control_dict["stats"].items():
                # Propagate global analysis settings
                cfg_with_global = {
                    "output_dir": self.output_dir,
                    "debug": self.debug,
                    **cfg,
                }
                needed_pairs = {k: self.paired[k] for k in cfg.get("data", []) if k in self.paired}
                compute_stats(needed_pairs, **cfg_with_global)

    def run(self):
        """
        Run the full evaluation workflow.
        If 'use_prefect' is enabled in the control file, it uses the Prefect/Dask
        orchestrator. Otherwise, it executes tasks sequentially in topological order.
        """
        use_prefect = self.control_dict["analysis"].get("use_prefect", False)

        if use_prefect:
            from melodies_monet.orchestrator.flows import main_orchestration_flow

            return main_orchestration_flow(self)
        else:
            order = self.graph_engine.get_execution_order()
            if self.debug:
                print(f"Execution order: {order}")

            # Sequential execution fallback
            # We process each interval if defined, otherwise run once
            intervals = self.time_intervals if self.time_intervals else [None]
            for interval in intervals:
                self.open_data(time_interval=interval)
                self.pair_data()
                self.stats()
                self.plotting()
