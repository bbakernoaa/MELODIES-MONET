# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import os
from melodies_monet.plots import savefig

mpl.use("Agg")

def test_savefig(tmpdir):
    save_dir = Path(tmpdir)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fp = save_dir / "asdf.png"

    # Currently must be str, not Path
    with pytest.raises(AttributeError, match="has no attribute 'split'"):
        savefig(fp)

    fname = fp.as_posix()
    savefig(fname)

    assert os.path.exists(fname)
    plt.close(fig)
