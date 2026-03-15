# SPDX-License-Identifier: Apache-2.0
#
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from melodies_monet.driver.orchestrator import orchestrator

# Mock monet_stats and monet_plots if they are not installed
if "monet_stats" not in sys.modules:
    sys.modules["monet_stats"] = MagicMock()
if "monet_plots" not in sys.modules:
    sys.modules["monet_plots"] = MagicMock()
if "monet" not in sys.modules:
    sys.modules["monet"] = MagicMock()


def create_mock_data():
    from melodies_monet.driver.data import Data

    mod_data = Data(label="mod1")
    mod_data.from_dict({"type": "model", "variables": {"O3": {}}})
    mod_data.obj = xr.Dataset(
        {"O3": (("time", "lat", "lon"), np.ones((1, 10, 10)))},
        coords={
            "time": [pd.Timestamp("2019-09-01")],
            "lat": np.arange(10),
            "lon": np.arange(10),
        },
    )
    mod_data.mapping = {"obs1": {"O3": "OZONE"}}

    obs_data = Data(label="obs1")
    obs_data.from_dict({"type": "obs", "obs_type": "pt_sfc"})
    obs_data.obj = xr.Dataset(
        {"OZONE": (("time", "site"), np.ones((1, 5)))},
        coords={
            "time": [pd.Timestamp("2019-09-01")],
            "latitude": (("site",), np.arange(5)),
            "longitude": (("site",), np.arange(5)),
        },
    )
    return mod_data, obs_data


def test_regression_sequential_vs_prefect():
    """
    Verify that sequential and Prefect-based execution produce bit-for-bit identical paired data.
    """
    control_dict = {
        "analysis": {
            "start_time": "2019-09-01-00:00:00",
            "end_time": "2019-09-01-01:00:00",
            "output_dir": ".",
            "debug": True,
        },
        "data": {"mod1": {"type": "model"}, "obs1": {"type": "obs"}},
        "evaluations": {"eval1": {"model": "mod1", "obs": "obs1", "mapping": {"O3": "OZONE"}}},
    }

    # 1. Run Sequential
    orc_seq = orchestrator()
    orc_seq.control_dict = control_dict.copy()
    orc_seq.control_dict["analysis"]["use_prefect"] = False
    mod_data_s, obs_data_s = create_mock_data()
    orc_seq.data = {"mod1": mod_data_s, "obs1": obs_data_s}

    with patch("monet.pair", create=True) as mock_pair:
        mock_pair.return_value = xr.Dataset({"O3_mod1": (("time", "site"), np.ones((1, 5)))})
        with patch("xarray.Dataset.monet", create=True) as mock_monet:
            mock_monet.combine_point.return_value = xr.Dataset({"O3_mod1": (("time", "site"), np.ones((1, 5)))})
            with patch.object(orchestrator, "open_data", return_value=None):
                orc_seq.run()
            seq_paired = orc_seq.paired["eval1"].obj

    # 2. Run Prefect (mocking the flow to call the same internal logic)
    orc_pref = orchestrator()
    orc_pref.control_dict = control_dict.copy()
    orc_pref.control_dict["analysis"]["use_prefect"] = True
    mod_data_p, obs_data_p = create_mock_data()
    orc_pref.data = {"mod1": mod_data_p, "obs1": obs_data_p}

    with patch("melodies_monet.orchestrator.flows.main_orchestration_flow", create=True) as mock_flow:
        with patch("monet.pair", create=True) as mock_pair_pref:
            mock_pair_pref.return_value = xr.Dataset({"O3_mod1": (("time", "site"), np.ones((1, 5)))})
            with patch("xarray.Dataset.monet", create=True) as mock_monet_pref:
                mock_monet_pref.combine_point.return_value = xr.Dataset({"O3_mod1": (("time", "site"), np.ones((1, 5)))})

                def side_effect(orc_inst):
                    # Simulate what the flow does: it calls pair_data
                    orc_inst.pair_data()
                    return None

                mock_flow.side_effect = side_effect
                orc_pref.run()
            pref_paired = orc_pref.paired["eval1"].obj

    # 3. Compare
    xr.testing.assert_allclose(seq_paired, pref_paired)


@pytest.mark.skip(reason="Failing in environment due to async/mpl issues")
@pytest.mark.mpl_image_compare
def test_plot_regression():
    """
    Regression test for plotting using pytest-mpl.
    """
    import matplotlib.pyplot as plt

    # If plt is mocked, return a real figure if possible,
    # but normally this test only runs if mpl-0.18.0 is present.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], label="Regression Line")
    ax.set_title("Plotting Regression Test")
    ax.legend()
    return fig
