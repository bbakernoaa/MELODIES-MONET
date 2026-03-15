# SPDX-License-Identifier: Apache-2.0
#
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["monet"] = MagicMock()
sys.modules["monetio"] = MagicMock()
sys.modules["monet_plots"] = MagicMock()
sys.modules["monet_stats"] = MagicMock()
sys.modules["xesmf"] = MagicMock()
sys.modules["cartopy"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

import xarray as xr

from melodies_monet.driver.analysis import analysis


def test_migrate_control_dict_unified_data():
    an = analysis()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-02-12:00:00",
            "end_time": "2019-08-03-12:00:00",
            "output_dir": "./output",
        },
        "data": {
            "cmaq_expt": {"type": "model", "files": "cmaq_files", "source": "cmaq"},
            "airnow": {"type": "obs", "files": "airnow_files", "obs_type": "pt_sfc"},
        },
        "evaluations": {
            "airnow_cmaq": {
                "model": "cmaq_expt",
                "obs": "airnow",
                "mapping": {"O3": "OZONE"},
            }
        },
    }
    an._migrate_control_dict()
    assert "data" in an.control_dict
    assert "cmaq_expt" in an.control_dict["data"]
    assert "airnow" in an.control_dict["data"]
    assert an.control_dict["data"]["cmaq_expt"]["source"] == "cmaq"


def test_migrate_control_dict_ref_exp():
    an = analysis()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-02-12:00:00",
            "end_time": "2019-08-03-12:00:00",
            "output_dir": "./output",
        },
        "data": {
            "cmaq_expt": {"type": "exp", "files": "cmaq_files", "source": "cmaq"},
            "airnow": {"type": "ref", "files": "airnow_files", "obs_type": "pt_sfc"},
        },
        "evaluations": {
            "airnow_cmaq": {
                "exp": "cmaq_expt",
                "ref": "airnow",
                "mapping": {"O3": "OZONE"},
            }
        },
    }
    an._migrate_control_dict()
    assert "data" in an.control_dict
    assert "cmaq_expt" in an.control_dict["data"]
    assert "airnow" in an.control_dict["data"]


def test_migrate_control_dict_legacy():
    an = analysis()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-02-12:00:00",
            "end_time": "2019-08-03-12:00:00",
            "output_dir": "./output",
        },
        "model": {
            "cmaq": {
                "mod_type": "cmaq",
                "files": "cmaq_files",
                "mapping": {"airnow": {"O3": "OZONE"}},
            }
        },
        "obs": {"airnow": {"obs_type": "pt_sfc", "filename": "airnow_file"}},
        "plots": {"plot_grp1": {"type": "timeseries"}},
    }
    an._migrate_control_dict()
    assert "data" in an.control_dict
    assert "plotting" in an.control_dict
    assert "evaluations" in an.control_dict
    assert "airnow_cmaq" in an.control_dict["evaluations"]
    assert an.control_dict["evaluations"]["airnow_cmaq"]["exp"] == "cmaq"
    assert an.control_dict["evaluations"]["airnow_cmaq"]["ref"] == "airnow"


def test_migrate_control_dict_list_eval():
    an = analysis()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-01",
            "end_time": "2019-08-02",
            "output_dir": ".",
        },
        "data": {
            "mod1": {"type": "exp"},
            "mod2": {"type": "exp"},
            "obs1": {"type": "ref"},
        },
        "evaluations": {
            "eval_grp": {
                "exp": ["mod1", "mod2"],
                "ref": "obs1",
                "mapping": {"O3": "OZONE"},
            }
        },
        "plotting": {"plot1": {"data": ["eval_grp"]}},
    }
    an._migrate_control_dict()
    assert "obs1_mod1" in an.control_dict["evaluations"]
    assert "obs1_mod2" in an.control_dict["evaluations"]
    assert "eval_grp" not in an.control_dict["evaluations"]
    assert an.control_dict["plotting"]["plot1"]["data"] == ["obs1_mod1", "obs1_mod2"]


def test_pair_data_mapping_logic():
    an = analysis()
    an.control_dict = {
        "analysis": {
            "start_time": "2019-08-01",
            "end_time": "2019-08-02",
            "output_dir": ".",
        },
        "evaluations": {"eval1": {"exp": "mod1", "ref": "obs1", "mapping": {"O3": "OZONE"}}},
    }
    # Mock models and obs
    mod = MagicMock()
    mod.label = "mod1"
    mod.mapping = None  # New schema style
    mod.variable_dict = {"O3": {}}
    mod.obj = MagicMock()

    obs = MagicMock()
    obs.label = "obs1"
    obs.obs_type = "pt_sfc"
    obs.obj = MagicMock()

    an.data = {"mod1": mod, "obs1": obs}

    # This should not crash
    from unittest.mock import patch

    with patch("monet.pair", create=True) as mock_pair:
        mock_pair.return_value = xr.Dataset()
        an.pair_data()
        assert "eval1" in an.paired
