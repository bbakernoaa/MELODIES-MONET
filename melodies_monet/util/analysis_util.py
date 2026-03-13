# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
from glob import glob


def fill_date_template(template_str, date_str):
    """
    Replace date template parameters with values from date string.

    Parameters
    ----------
    template_str : str
        Template string.
    date_str : str
        Date string (yyyy-mm-m_abbr-dd-ddd).

    Returns
    -------
    str
        Filled template string.
    """

    yyyy_str, mm_str, m_abbr_str, dd_str, ddd_str = tuple(date_str.split("-"))

    if "DDD" in template_str:
        return template_str.replace("YYYY", yyyy_str).replace("DDD", ddd_str)
    else:
        return template_str.replace("YYYY", yyyy_str).replace("MM", mm_str).replace("M_ABBR", m_abbr_str).replace("DD", dd_str)


def find_file(datadir, filestr):
    """
    Find a file matching a pattern in a directory.

    Parameters
    ----------
    datadir : str
        Data directory.
    filestr : str
        Filename regular expression.

    Returns
    -------
    str
        Complete path of matching filename in data directory.
    """
    logger = logging.getLogger(__name__)

    pattern = os.path.join(os.path.expandvars(datadir), filestr)
    files = glob(pattern)

    if len(files) == 0:
        raise Exception("no file matches for %s" % pattern)
    if len(files) > 1:
        raise Exception("more than one file match %s" % pattern)

    filename = files[0]
    logger.info(filename)

    return filename


def get_obs_vars(config):
    """
    Get subset of obs variables from model to obs variable mapping.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Nested dictionary keyed by obs set name and obs variable name.
    """
    obs_vars_subset = dict()

    for model_name in config["model"]:
        mapping = config["model"][model_name]["mapping"]

        for obs_name in mapping:
            obs_vars = config["obs"][obs_name]["variables"]
            obs_vars_subset[obs_name] = dict()

            for model_var in mapping[obs_name]:
                obs_var = mapping[obs_name][model_var]
                obs_vars_subset[obs_name][obs_var] = obs_vars[obs_var]

    return obs_vars_subset
