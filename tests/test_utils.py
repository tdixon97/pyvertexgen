from __future__ import annotations

import legendhpges
import numpy as np
import pytest
from legendtestdata import LegendTestData
from pyg4ometry import geant4
from lgdo import lh5

from pyvertexgen.utils import save_vertex_lh5
import os
import awkward as ak

@pytest.fixture
def temp_file(tmp_path):
    return str(tmp_path)


def test_write(temp_file):

    # fake processor
    def proc(n=100,seed=1):
        rng = np.random.default_rng(seed)
        return rng.uniform(low=-50,high=50,size=(n,3))
    
    save_vertex_lh5(temp_file+"test.lh5",10000,proc,first_seed=1,buffer=100)

    # test file exists
    assert os.path.exists(temp_file+"test.lh5")    
    vertices = lh5.read("vertices",temp_file+"test.lh5").view_as("ak")

    assert len(vertices.xpos)==10000

    # now with total not divided by the buffer
    save_vertex_lh5(temp_file+"test.lh5",10300,proc,first_seed=1,buffer=100)

    # test file exists
    assert os.path.exists(temp_file+"test.lh5")    
    vertices = lh5.read("vertices",temp_file+"test.lh5").view_as("ak")
    assert len(vertices.xpos)==10300
