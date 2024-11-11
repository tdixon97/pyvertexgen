from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from lgdo import Array, Table, lh5
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def save_vertex_lh5(
    file_out: str,
    n_vertex: int,
    generator: Callable[[int, int], ArrayLike],
    first_seed: int | None = None,
    buffer: int = int(1e6),
) -> None:
    """Function for IO of vertices to an lh5 file.

    Parameters
    ----------
    file_out
        output file
    n_vertex
        number of vertices to generate.
    generator
        python function taking two arguments `n` and `seed` and returning the vertices. More complicated functions should be wrapped before being passed.
    first_seed
        First random seed for the rng (if None no seeds are used). This seed is used for the first chunk of files and one is added for each subsequent chunk.
    buffer
        buffer size for LH5 i/o.
    """

    # get number of events per chunk
    n_lists = [buffer]
    n_tot = buffer

    while n_tot < n_vertex:
        n_lists.append(buffer)
        n_tot += buffer

    n_lists[-1] -= np.sum(n_lists) - n_vertex

    for idx, n in enumerate(n_lists):
        mode = "of" if (idx == 0) else "append"
        seed_tmp = first_seed + idx if (first_seed is not None) else None
        vertices = generator(n=n, seed=seed_tmp)

        # turn into an lh5 table
        out_tbl = Table(size=n)

        out_tbl.add_field("xpos", Array(vertices[:, 0]))
        out_tbl.add_field("ypos", Array(vertices[:, 1]))
        out_tbl.add_field("zpos", Array(vertices[:, 2]))

        lh5.write(out_tbl, name="vertices", lh5_file=file_out, wo_mode=mode)
