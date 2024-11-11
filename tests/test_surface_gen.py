from __future__ import annotations

import argparse
import logging

import colorlog
import pytest
import legendhpges
import numpy as np
from legendtestdata import LegendTestData
from pyvertexgen.generators import generate_hpge_surface,generate_many_hpge_surface
from pyg4ometry import geant4

@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")



def test_surface_gen(test_data_configs):

    hpge = legendhpges.make_hpge(test_data_configs + "/V99000A.json")

    coords = generate_hpge_surface(100,hpge,surface_type=None,depth=None)
    assert np.shape(coords)==(100,3)
    
    dist = hpge.distance_to_surface(coords)
    assert np.allclose(a=dist,b=(1e-11)*np.ones_like(dist),atol=1e-9)

    # test one surf type
    coords = generate_hpge_surface(100,hpge,surface_type="pplus",depth=None)
    assert np.allclose(a=dist,b=(1e-11)*np.ones_like(dist),atol=1e-9)

    coords = generate_hpge_surface(100,hpge,surface_type="nplus",depth=None)
    assert np.allclose(a=dist,b=(1e-11)*np.ones_like(dist),atol=1e-9)

    coords = generate_hpge_surface(100,hpge,surface_type="passive",depth=None)
    assert np.allclose(a=dist,b=(1e-11)*np.ones_like(dist),atol=1e-9)

def test_many_surface_gen(test_data_configs):

    reg =geant4.Registry()
    hpge_IC = legendhpges.make_hpge(test_data_configs + "/V99000A.json",registry=reg)
    hpge_BG = legendhpges.make_hpge(test_data_configs + "/B99000A.json",registry=reg)
    hpge_SC = legendhpges.make_hpge(test_data_configs + "/C99000A.json",registry=reg)

    coords,ids = generate_many_hpge_surface(1000,[hpge_IC,hpge_BG,hpge_SC])

    assert np.shape(coords)==(1000,3)
    assert np.shape(ids)==(1000,)