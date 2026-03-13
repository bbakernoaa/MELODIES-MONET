from dask.distributed import worker_client
from prefect import get_run_logger, task

import monet as m
from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair as pair_inst


@task
def run_data_task(data_type, label, cfg, time_interval=None, **kwargs):
    """
    Task to load data (model or observation).

    Parameters
    ----------
    data_type : str
        'model' or 'obs'.
    label : str
        Unique label for the data source.
    cfg : dict
        Configuration dictionary for the data source.
    time_interval : list of pd.Timestamp, optional
        A list containing [start_time, end_time] to subset the data.

    Returns
    -------
    Data
        The loaded Data instance.
    """
    logger = get_run_logger()
    logger.info(f"Loading {data_type} data: {label}")
    # cfg already contains 'type' due to migration or unified schema
    inst = Data(data_type=data_type)
    inst.label = label
    inst.from_dict(cfg)
    inst.load(time_interval=time_interval)
    return inst


@task
def run_pairing_task(eval_label, cfg, model_inst, obs_inst, pairing_kwargs, **kwargs):
    """
    Task to pair model data with observation data.

    Parameters
    ----------
    eval_label : str
        Unique label for the evaluation/pairing.
    cfg : dict
        Configuration dictionary for the evaluation.
    model_inst : Data
        The model Data instance.
    obs_inst : Data
        The observation Data instance.
    pairing_kwargs : dict
        Global pairing keyword arguments.

    Returns
    -------
    pair
        The resulting pair instance containing paired data.
    """
    logger = get_run_logger()
    logger.info(f"Pairing evaluation: {eval_label}")
    mapping = cfg.get("mapping")
    if mapping is None:
        mapping = model_inst.mapping.get(obs_inst.label, {}) if model_inst.mapping else {}
    keys = list(mapping.keys())
    ref_vars = list(mapping.values())
    mod_vars = list(model_inst.variable_dict.keys()) if model_inst.variable_dict else []

    with worker_client() as client:
        futures = []
        for var_mod, var_obs in mapping.items():
            vars_to_pair = list(set([var_mod] + mod_vars))
            f = client.submit(
                m.pair,
                model_inst.obj[vars_to_pair],
                obs_inst.obj,
                radius_of_influence=model_inst.radius_of_influence,
                suffix=model_inst.label,
                type=obs_inst.obs_type.lower(),
                **pairing_kwargs.get(obs_inst.obs_type.lower(), {}),
            )
            futures.append(f)
        results = client.gather(futures)
        if len(results) > 1:
            import xarray as xr

            paired_data = xr.merge(results)
        else:
            paired_data = results[0]

    p_inst = pair_inst()
    p_inst.ref = obs_inst.label
    p_inst.model = model_inst.label
    p_inst.model_vars = keys
    p_inst.ref_vars = ref_vars
    p_inst.type = obs_inst.obs_type.lower()
    p_inst.obj = paired_data
    return p_inst


@task
def run_stats_task(group_label, cfg, paired_dict, analysis_cfg=None, **kwargs):
    """
    Task to calculate statistics for a group of pairings.

    Parameters
    ----------
    group_label : str
        Label for the statistics group.
    cfg : dict
        Configuration dictionary for the statistics group.
    paired_dict : dict
        Dictionary of available pair instances.
    analysis_cfg : dict, optional
        Global analysis configuration for propagation.

    Returns
    -------
    object
        The statistics results object.
    """
    logger = get_run_logger()
    logger.info(f"Calculating stats for group: {group_label}")

    try:
        import monet_stats
    except ImportError:
        logger.error("monet_stats not found. Statistics cannot be calculated.")
        return f"Stats for {group_label} failed: monet_stats not found"

    # Propagate global analysis settings
    analysis_cfg = analysis_cfg or {}
    cfg_with_global = {
        "output_dir": analysis_cfg.get("output_dir"),
        "debug": analysis_cfg.get("debug"),
        **cfg,
    }

    needed_pairs = {k: paired_dict[k] for k in cfg.get("data", []) if k in paired_dict}
    return monet_stats.compute_stats(needed_pairs, **cfg_with_global)


@task
def run_plotting_task(group_label, cfg, paired_dict, analysis_cfg=None, **kwargs):
    """
    Task to generate plots for a group of pairings.

    Parameters
    ----------
    group_label : str
        Label for the plotting group.
    cfg : dict
        Configuration dictionary for the plotting group.
    paired_dict : dict
        Dictionary of available pair instances.
    analysis_cfg : dict, optional
        Global analysis configuration for propagation.

    Returns
    -------
    object
        The plotting results object.
    """
    logger = get_run_logger()
    logger.info(f"Generating plots for group: {group_label}")

    try:
        import monet_plots
    except ImportError:
        logger.error("monet_plots not found. Plotting cannot be performed.")
        return f"Plotting for {group_label} failed: monet_plots not found"

    # Propagate global analysis settings
    analysis_cfg = analysis_cfg or {}
    cfg_with_global = {
        "output_dir": analysis_cfg.get("output_dir"),
        "debug": analysis_cfg.get("debug"),
        "add_logo": analysis_cfg.get("add_logo", True),
        **cfg,
    }

    needed_pairs = {k: paired_dict[k] for k in cfg.get("data", []) if k in paired_dict}
    return monet_plots.create_plots(needed_pairs, **cfg_with_global)
