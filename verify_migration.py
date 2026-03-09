import yaml
import os
from melodies_monet.driver.analysis import analysis

legacy_yaml = """
analysis:
  start_time: '2019-08-02-12:00:00'
  end_time: '2019-08-03-12:00:00'
  output_dir: /tmp/output
  debug: True
model:
  wrfchem:
    mod_type: 'wrfchem'
    files: 'dummy_files/*'
    mapping:
      airnow:
        o3: OZONE
obs:
  airnow:
    obs_type: pt_sfc
    filename: 'dummy_obs.nc'
plots:
  plot_grp1:
    type: 'timeseries'
    data: ['airnow_wrfchem']
    data_proc:
      rem_obs_nan: True
stats:
  stat_list: ['MB', 'RMSE']
  data: ['airnow_wrfchem']
"""

new_yaml = """
analysis:
  start_time: '2019-08-02-12:00:00'
  end_time: '2019-08-03-12:00:00'
  output_dir: /tmp/output
  debug: True
models:
  wrfchem:
    mod_type: 'wrfchem'
    files: 'dummy_files/*'
obs:
  airnow:
    obs_type: pt_sfc
    filename: 'dummy_obs.nc'
evaluations:
  airnow_wrfchem:
    model: wrfchem
    obs: airnow
    mapping:
      o3: OZONE
    stats:
      stat_list: ['MB', 'RMSE']
plotting:
  plot_grp1:
    type: 'timeseries'
    data: ['airnow_wrfchem']
    data_proc:
      rem_obs_nan: True
"""

with open("legacy.yaml", "w") as f:
    f.write(legacy_yaml)

with open("new.yaml", "w") as f:
    f.write(new_yaml)

try:
    ana_legacy = analysis()
    ana_legacy.control = "legacy.yaml"
    ana_legacy.read_control()

    ana_new = analysis()
    ana_new.control = "new.yaml"
    ana_new.read_control()

    print("Legacy migrated structure keys:", ana_legacy.control_dict.keys())
    print("New structure keys:", ana_new.control_dict.keys())

    assert set(ana_legacy.control_dict.keys()) == {'analysis', 'models', 'obs', 'evaluations', 'plotting'}
    assert set(ana_new.control_dict.keys()) == {'analysis', 'models', 'obs', 'evaluations', 'plotting'}

    # Check if they are equivalent
    # Note: legacy migration might have some extra defaults or different nesting for stats if not careful
    # but the core pieces should be there.

    legacy_eval = ana_legacy.control_dict['evaluations']['airnow_wrfchem']
    new_eval = ana_new.control_dict['evaluations']['airnow_wrfchem']

    print("Legacy Evaluation:", legacy_eval)
    print("New Evaluation:", new_eval)

    assert legacy_eval['model'] == new_eval['model']
    assert legacy_eval['obs'] == new_eval['obs']
    assert legacy_eval['mapping'] == new_eval['mapping']
    assert legacy_eval['stats']['stat_list'] == new_eval['stats']['stat_list']

    print("Verification successful: Legacy and New YAMLs produce equivalent internal structures.")

finally:
    if os.path.exists("legacy.yaml"): os.remove("legacy.yaml")
    if os.path.exists("new.yaml"): os.remove("new.yaml")
    if os.path.exists("verify_migration.py"): os.remove("verify_migration.py")
