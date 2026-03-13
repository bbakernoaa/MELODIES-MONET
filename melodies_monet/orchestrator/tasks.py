from prefect import task, get_run_logger
from dask.distributed import worker_client
import monet as m
from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair as pair_inst

@task
def run_data_task(data_type, label, cfg, time_interval=None, **kwargs):
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
                **pairing_kwargs.get(obs_inst.obs_type.lower(), {})
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
def run_stats_task(group_label, cfg, paired_dict, **kwargs):
    logger = get_run_logger()
    logger.info(f"Calculating stats for group: {group_label}")
    needed_pairs = {k: paired_dict[k] for k in cfg.get("data", []) if k in paired_dict}

    from melodies_monet import stats
    # Real implementation: return stats.compute_stats(needed_pairs, **cfg)
    return f"Stats for {group_label} completed"

@task
def run_plotting_task(group_label, cfg, paired_dict, **kwargs):
    logger = get_run_logger()
    logger.info(f"Generating plots for group: {group_label}")
    needed_pairs = {k: paired_dict[k] for k in cfg.get("data", []) if k in paired_dict}

    from melodies_monet import plots
    # Real implementation: return plots.create_plots(needed_pairs, **cfg)
    return f"Plotting for {group_label} completed"
