# SPDX-License-Identifier: Apache-2.0
#
"""
Plotting routines.
"""

from functools import partial
from pathlib import Path

from monet import savefig as monet_savefig

LOGO_PATH = Path(__file__).parent / "../data/MM_logo.png"


def savefig(fname, **kwargs):
    """
    Wrapper around monet.savefig with MELODIES-MONET branding.
    """
    if isinstance(fname, Path):
        fname = str(fname)
    # Ensure directory exists before calling monet.savefig
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    return monet_savefig(fname, logo=LOGO_PATH, loc=2, decorate=True, bbox_inches="tight", dpi=200, **kwargs)


__all__ = ("savefig",)
