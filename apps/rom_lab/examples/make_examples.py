"""Regenerate the two example datasets of the ROM Lab from PINNeAPPle's own solvers (takes ~10 minutes on 4 cores).

    python apps/rom_lab/examples/make_examples.py

* ``cylinder_wake_re100.npz``: lattice-Boltzmann flow past a cylinder at Re = 100 (D = 20 cells), 30,000 steps, a
  snapshot every 50; the periodic part (every second snapshot, every fourth cell) as ux, uy on 60 x 25 points.
  ``hf_seconds``: the measured LBM time to advance from one stored snapshot to the next.
* ``cantilever_designs.npz``: 100 designs of a 3-D solid FEM cantilever (height, width, load angle; Latin hypercube),
  nodal displacement and von Mises stress. ``hf_seconds``: the measured FEM time per design.
"""
from __future__ import annotations

import os
import time

import numpy as np

from pinneapple_data.snapshots import snapshots_from_arrays
from pinneapple_lab.experiments.rom import cantilever, cylinder_wake
from pinneapple_neural.architectures.rom import latin_hypercube

HERE = os.path.dirname(os.path.abspath(__file__))


def wake():
    t0 = time.perf_counter()
    steps, every, D = 30000, 50, 20
    ux, uy, solid, g = cylinder_wake(100.0, steps, every, D=D)
    secs = time.perf_counter() - t0
    keep = np.arange(len(ux) - 300, len(ux), 2)                  # periodic regime, every second snapshot
    sub = (slice(None), slice(None, None, 4), slice(None, None, 4))
    ux, uy = ux[keep][sub], uy[keep][sub]
    nx, ny = ux.shape[1:]
    xs, ys = np.arange(0, g["nx"], 4)[:nx], np.arange(0, g["ny"], 4)[:ny]
    P = np.stack(np.meshgrid((xs - g["cx"]) / D, (ys - g["cy"]) / D, indexing="ij"), -1).reshape(-1, 2)
    U = np.stack([ux.reshape(len(keep), -1), uy.reshape(len(keep), -1)], -1) / g["u_in"]
    t = keep * every * g["u_in"] / D                             # convective time units D / U
    s = snapshots_from_arrays({"U": U.astype(np.float32)}, times=t, locations={"point": np.c_[P, np.zeros(len(P))]},
                              meta={"hf_seconds": secs / (steps / every) * 2, "dt": float(t[1] - t[0]),
                                    "units": {"U": "U_inf", "x": "D", "t": "D/U_inf"}})
    open(os.path.join(HERE, "cylinder_wake_re100.npz"), "wb").write(s.to_npz_bytes())
    print("wake", s.summary()["n_snapshots"], s.n_values, f"{secs:.0f} s")


def beam():
    box = {"H": (0.12, 0.3), "W": (0.06, 0.2), "angle_deg": (0.0, 90.0)}
    P, names = latin_hypercube(100, box, 0)
    t0 = time.perf_counter()
    R = [cantilever(*q) for q in P]
    secs = (time.perf_counter() - t0) / len(P)
    s = snapshots_from_arrays({"displacement": np.array([r.u for r in R], np.float32) * 1e3,
                               "von_mises": np.array([r.von_mises for r in R], np.float32) / 1e6},
                              params=P, param_names=names, locations={"point": R[0].mesh.nodes},
                              meta={"hf_seconds": secs, "units": {"displacement": "mm", "von_mises": "MPa"}})
    open(os.path.join(HERE, "cantilever_designs.npz"), "wb").write(s.to_npz_bytes())
    print("beam", s.n, s.n_values, f"{secs:.2f} s per design")


if __name__ == "__main__":
    beam()
    wake()
