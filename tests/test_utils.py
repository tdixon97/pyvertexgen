from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lgdo import lh5
import h5py
from pyvertexgen.utils import save_vertex_lh5
import awkward as ak

@pytest.fixture
def temp_file(tmp_path):
    return str(tmp_path)


def test_write(temp_file):
    # fake processor
    def proc(n=100, seed=1):
        rng = np.random.default_rng(seed)
        return rng.uniform(low=-50, high=50, size=(n, 3))

    save_vertex_lh5(temp_file + "test.lh5", 10000, proc, first_seed=1, buffer=100)

    # test file exists
    assert Path.exists(Path(temp_file + "test.lh5"))
    vertices = lh5.read("vertices", temp_file + "test.lh5").view_as("ak")

    assert len(vertices.xpos) == 10000

    # now with total not divided by the buffer
    save_vertex_lh5(temp_file + "test.lh5", 10300, proc, first_seed=1, buffer=100)

    # test file exists
    assert Path.exists(Path(temp_file + "test.lh5"))

    vertices = lh5.read("vertices", temp_file + "test.lh5").view_as("ak")
    assert len(vertices.xpos) == 10300

    # try writing to hdf5
    save_vertex_lh5(temp_file + "test.h5", 10300, proc, first_seed=1, buffer=100,file_type="hdf5")

    assert Path.exists(Path(temp_file + "test.h5"))

    # read it back
    with h5py.File(Path(temp_file+"test.h5"), 'r') as f:
        # Access the dataset
        vertices = f['vertices']
        
        # Read data from each column
        xpos = vertices['xpos'][:]
        ypos = vertices['ypos'][:]
        zpos = vertices['zpos'][:]

        vertices= ak.Array({"xpos":xpos,"ypos":ypos,"zpos":zpos})
    
    assert len(vertices.ypos) == 10300
    assert len(vertices.xpos) == 10300
    assert len(vertices.zpos) == 10300
