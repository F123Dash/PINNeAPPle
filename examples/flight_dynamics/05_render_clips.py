"""Blender Cycles clips of the take-off and of the severe-turbulence encounter of every aircraft, with the time and
the main state (airspeed, height, n_z, pitch, bank) burnt in, from the runs saved by 03_takeoff.py and
04_turbulence_encounters.py (run those first).

    python examples/flight_dynamics/05_render_clips.py [--frames 96] [--size 720x405] [--samples 12] [--only a320]

Writes MP4 clips (H.264) and a small GIF per clip into docs/assets/flight_dynamics/clips/ and one still per clip.
Full-resolution renders: raise --size / --samples / --frames (outputs then stay in _out/, not in the repository).
"""
import argparse
import os
import pickle
import shutil

from pinneapple_simulation.flight import catalog
from pinneapple_simulation.flight.visuals import flight_scene
from pinneapple_tools.visualization.studio.blender import render_animation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_out")
CLIPS = os.path.join(HERE, "..", "..", "docs", "assets", "flight_dynamics", "clips")

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=96)
ap.add_argument("--size", default="720x405")
ap.add_argument("--samples", type=int, default=12)
ap.add_argument("--only", default=None)
ap.add_argument("--keep-in-out", action="store_true", help="do not copy the clips into docs/assets")
a = ap.parse_args()
size = tuple(int(v) for v in a.size.split("x"))
os.makedirs(CLIPS, exist_ok=True)
for name in catalog.names():
    if a.only and name != a.only:
        continue
    ac = catalog.build(name)
    for case, camera, runway in (("takeoff", "side", True), ("turbulence", "chase", False)):
        src = os.path.join(OUT, f"{name}_{case}", "run.pkl")
        if not os.path.exists(src):
            print("missing", src)
            continue
        run = pickle.load(open(src, "rb"))
        title = f"{ac.info['type']}: {'take-off' if case == 'takeoff' else 'severe turbulence'}"
        sc = flight_scene(ac, run["hist"], n_frames=a.frames, wind=run["wind"] if case != "takeoff" else None,
                          runway=runway, title=title)
        out_dir = os.path.join(OUT, f"{name}_{case}", "frames")
        r = render_animation(sc, out_dir, size=size, samples=a.samples, camera=camera, trails=False,
                             fps=12, gif=True, mp4=True, name=f"{name}_{case}", ground=None if runway else False)
        print(name, case, r["mp4"], r["gif"])
        if not a.keep_in_out:
            for k in ("mp4", "gif"):
                if r[k]:
                    shutil.copy(r[k], os.path.join(CLIPS, os.path.basename(r[k])))
            shutil.copy(r["frames"][len(r["frames"]) * 2 // 3], os.path.join(CLIPS, f"{name}_{case}.png"))
