# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import os
from melodies_monet.plots import savefig

mpl.use("Agg")

def test_savefig(tmp_path):
    print(f"\ntmp_path: {tmp_path}")
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    fp = tmp_path / "test_fig.png"
    print(f"fp: {fp}")

    # Test with Path object
    savefig(fp)
    print(f"Checking if {fp} exists...")
    exists = fp.exists()
    print(f"Exists: {exists}")
    if not exists:
        print(f"Directory contents: {os.listdir(tmp_path)}")
    assert exists

    # Test with string
    fp2 = (tmp_path / "test_fig2.png").as_posix()
    savefig(fp2)
    assert Path(fp2).exists()

    plt.close(fig)
