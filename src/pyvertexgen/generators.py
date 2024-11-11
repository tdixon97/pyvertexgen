from __future__ import annotations

import argparse
import logging

import colorlog

import legendhpges

from numpy.typing import NDArray
from scipy.stats import rv_continuous
import numpy as np
from typing import Tuple,List
from numba import njit
log = logging.getLogger(__name__)


def generate_many_hpge_surface(n_tot:int,hpges:list[legendhpges.HPGe],surface_type:str|None=None)->Tuple[NDArray,NDArray]:
    """Generate events on many HPGe's weighting by the surface area.

    Parameters
    ----------
    n_tot
        total number of events to generate
    hpges
        List of :class:`legendhpges.HPGe` objects.
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).

    Returns
    -------
    (local_coords,det_ids) 
        tuple of an NDArray of local coordinates and another of detector_ids (index of the hpges list).
    """

    out = np.full((n_tot,3),np.nan)

    # index of the surfaces per detector
    surf_ids_tot = [np.array(hpge.surfaces)==surface_type if surface_type is not None else np.arange(len(hpge.surfaces)) for hpge in hpges]

    # total surface area per detector
    surf_tot =[ np.sum(hpge.surface_area(surf_ids).magnitude) for  hpge,surf_ids in zip(hpges,surf_ids_tot)]
    print(surf_tot)

    p_det = surf_tot/np.sum(surf_tot)

    det_index = np.random.choice(np.arange(len(hpges)),size=n_tot,p=p_det)

    # loop over n_det maybe could be faster
    for idx, hpge in enumerate(hpges):
        n = np.sum(det_index==idx)
        out[det_index==idx]=generate_hpge_surface(n,hpge,surface_type=surface_type)

    return out,det_index


def generate_hpge_surface(n:int,hpge:legendhpges.HPGe,surface_type:str|None,depth:rv_continous| None=None)-> NDArray:
    """Generate events on the surface of the HPGe.

    Parameters
    ----------
    n
        number of vertexs to generate.
    hpge
        legendhpges object describing the detector geometry.
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).
    depth
        scipy `rv_continuous` object describing the depth profile, if None events are generated directly on the surface.

    Returns
    -------
    NDArray with shape `(n,3)` descirbing the local `(x,y,z)` positions for every vertex
    """

    # get the indices (r,z) pairs
   
    surf=np.array(hpge.surfaces)
    surface_indices = np.where(surf == surface_type)[0] if (surface_type is not None) else np.arange(len(hpge.surfaces))
    
    # surface areas
    areas = hpge.surface_area(surface_indices).magnitude

    # get the sides
    sides = np.random.choice(surface_indices,size=n,p=areas/np.sum(areas))
    # get thhe detector geometry
    r,z = hpge.get_profile()
    s1,s2 = legendhpges.utils.get_line_segments(r,z)

    # compute random coordinates
    frac = np.random.uniform(low=0,high=1,size=(len(sides)))
    rz_coords = s1[sides]+(s2[sides]-s1[sides])*frac[:,np.newaxis]
    
    phi = np.random.uniform(low=0,high=2*np.pi,size=(len(sides)))
    
    # convert to random x,y
    x = rz_coords[:,0]*np.cos(phi)
    y = rz_coords[:,0]*np.sin(phi)

    if (depth is not None):
        raise NotImplementedError("depth profile is not yet implemented ")
    
    return np.vstack([x,y,rz_coords[:,1]]).T