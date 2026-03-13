import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules['monet'] = MagicMock()
sys.modules['monetio'] = MagicMock()
sys.modules['monet_plots'] = MagicMock()
sys.modules['monet_stats'] = MagicMock()
sys.modules['xesmf'] = MagicMock()
sys.modules['cartopy'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import pytest
import yaml
from melodies_monet.driver.analysis import analysis

def test_migrate_control_dict_unified_data():
    an = analysis()
    an.control_dict = {
        'analysis': {
            'start_time': '2019-08-02-12:00:00',
            'end_time': '2019-08-03-12:00:00',
            'output_dir': './output'
        },
        'data': {
            'cmaq_expt': {
                'type': 'model',
                'files': 'cmaq_files',
                'source': 'cmaq'
            },
            'airnow': {
                'type': 'obs',
                'files': 'airnow_files',
                'obs_type': 'pt_sfc'
            }
        },
        'evaluations': {
            'airnow_cmaq': {
                'model': 'cmaq_expt',
                'obs': 'airnow',
                'mapping': {'O3': 'OZONE'}
            }
        }
    }
    an._migrate_control_dict()
    assert 'models' in an.control_dict
    assert 'obs' in an.control_dict
    assert 'cmaq_expt' in an.control_dict['models']
    assert 'airnow' in an.control_dict['obs']
    assert an.control_dict['models']['cmaq_expt']['source'] == 'cmaq'

def test_migrate_control_dict_legacy():
    an = analysis()
    an.control_dict = {
        'analysis': {
            'start_time': '2019-08-02-12:00:00',
            'end_time': '2019-08-03-12:00:00',
            'output_dir': './output'
        },
        'model': {
            'cmaq': {
                'mod_type': 'cmaq',
                'files': 'cmaq_files',
                'mapping': {
                    'airnow': {'O3': 'OZONE'}
                }
            }
        },
        'obs': {
            'airnow': {
                'obs_type': 'pt_sfc',
                'filename': 'airnow_file'
            }
        },
        'plots': {
            'plot_grp1': {'type': 'timeseries'}
        }
    }
    an._migrate_control_dict()
    assert 'models' in an.control_dict
    assert 'plotting' in an.control_dict
    assert 'evaluations' in an.control_dict
    assert 'airnow_cmaq' in an.control_dict['evaluations']
    assert an.control_dict['evaluations']['airnow_cmaq']['model'] == 'cmaq'
    assert an.control_dict['evaluations']['airnow_cmaq']['obs'] == 'airnow'

def test_pair_data_mapping_logic():
    an = analysis()
    an.control_dict = {
        'analysis': {'start_time': '2019-08-01', 'end_time': '2019-08-02', 'output_dir': '.'},
        'evaluations': {
            'eval1': {
                'model': 'mod1',
                'obs': 'obs1',
                'mapping': {'O3': 'OZONE'}
            }
        }
    }
    # Mock models and obs
    mod = MagicMock()
    mod.label = 'mod1'
    mod.mapping = None  # New schema style
    mod.variable_dict = {'O3': {}}
    mod.obj = MagicMock()

    obs = MagicMock()
    obs.label = 'obs1'
    obs.obs_type = 'pt_sfc'
    obs.obj = MagicMock()

    an.models = {'mod1': mod}
    an.obs = {'obs1': obs}

    # This should not crash
    from unittest.mock import patch
    with patch('monet.pair') as mock_pair:
        an.pair_data()
        assert 'eval1' in an.paired
        # Check mapping usage
        # In pair_data: keys = list(mapping.keys())
        # model_obj = mod.obj[list(set(keys + mod_vars))]
        # We can't easily check internal calls without more complex mocks,
        # but the lack of crash is already a good sign.
