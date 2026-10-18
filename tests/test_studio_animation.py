"""Time animations of studio scenes: Animation data, poses, scene.json, the browser viewer and the Blender renderer."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pinneapple_tools.visualization.studio import Scene, Surface
from pinneapple_tools.visualization.studio.animation import (
    Animation,
    matrix_from_quat,
    quat_from_matrix,
    slerp,
)


def _cube(s=1.0):
    V = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], float) / 2
    F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return V, np.array(F)


def _yaw(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _scene(T=6):
    V, F = _cube()
    sc = Scene([Surface("floor", [[-5, -5, -1], [5, -5, -1], [5, 5, -1], [-5, 5, -1]], [[0, 1, 2], [0, 2, 3]],
                        "concrete"),
                Surface("cube", V, F, "copper", group="body")])
    t = np.linspace(0, 1, T)
    pos = np.c_[3 * np.cos(np.pi * t), 3 * np.sin(np.pi * t), 0 * t]
    q = quat_from_matrix(np.array([_yaw(np.pi * tt + np.pi / 2) for tt in t]))
    an = Animation(t).add_track("body", pos, q)
    o = np.zeros((T, 1, 3))
    v = np.tile([1.5, 0.0, 0.0], (T, 1, 1))
    an.add_vectors("thrust", o, v, values=np.linspace(0, 1, T)[:, None], attach="body", label="Thrust [kN]")
    an.add_trail("path", pos)
    an.add_surface_field("cube", np.outer(np.linspace(-1, 1, T), V[:, 0]), label="dCp")
    an.add_hud("V", 10 * t, "Speed", "m/s")
    sc.animation = an
    return sc


def test_quaternion_matrix_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    R = matrix_from_quat(q)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1)
    assert np.allclose(quat_from_matrix(R), q, atol=1e-12)
    assert np.allclose(matrix_from_quat(quat_from_matrix(_yaw(0.7))), _yaw(0.7))


def test_slerp_unit_norm_and_endpoints():
    q0 = quat_from_matrix(_yaw(0.1))
    q1 = quat_from_matrix(_yaw(2.0))
    assert np.allclose(slerp(q0, q1, 0), q0) and np.allclose(slerp(q0, q1, 1), q1)
    for f in np.linspace(0, 1, 11):
        assert abs(np.linalg.norm(slerp(q0, q1, f)) - 1) < 1e-12
    # half way is the rotation by the mean angle
    assert np.allclose(matrix_from_quat(slerp(q0, q1, 0.5)), _yaw(1.05))


def test_pose_interpolation_endpoints_exact():
    an = _scene().animation
    for k in (0, 3, 5):
        assert np.allclose(an.at(an.times[k], "body"), an.pose("body", k))
    M = an.at(0.5 * (an.times[1] + an.times[2]), "body")
    assert np.allclose(M[:3, 3], 0.5 * (an.tracks["body"]["positions"][1] + an.tracks["body"]["positions"][2]))
    assert np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3))
    assert set(an.frame(2)) == {"body"}


def test_body_frame_vertex_lands_in_world():
    an = _scene().animation
    k = 5                                                 # t = 1: at (-3, 0, 0), yawed by 270 degrees
    p = an.transform("body", k, [[1.0, 0.0, 0.0]])[0]      # nose of the body: +x in the body frame
    assert np.allclose(p, [-3.0, 0.0, 0.0] + _yaw(1.5 * np.pi) @ [1, 0, 0], atol=1e-9)
    o, v = an.vectors_world(0, k)                          # attached arrow turns with the body
    assert np.allclose(o[0], [-3, 0, 0], atol=1e-9) and np.allclose(v[0], [0, -1.5, 0], atol=1e-9)


def test_dict_round_trip_and_resample():
    an = _scene().animation
    d = json.loads(json.dumps(an.to_dict()))
    b = Animation.from_dict(d)
    assert np.allclose(b.times, an.times)
    assert np.allclose(b.tracks["body"]["positions"], an.tracks["body"]["positions"], atol=1e-4)
    assert np.allclose(b.tracks["body"]["quaternions"], an.tracks["body"]["quaternions"], atol=1e-6)
    assert np.allclose(b.vectors[0]["values"], an.vectors[0]["values"], atol=1e-4)
    assert b.hud[0]["label"] == "Speed" and d["hud"][0]["nd"] == 1
    r = an.resample(11)
    assert r.n_frames == 11 and np.allclose(r.times[[0, -1]], an.times[[0, -1]])
    assert np.allclose(r.pose("body", 0), an.pose("body", 0)) and np.allclose(r.pose("body", 10), an.pose("body", 5))
    assert np.allclose(np.linalg.norm(r.tracks["body"]["quaternions"], axis=1), 1)
    assert r.surface_fields["cube"]["values"].shape == (11, 8)


def test_shape_errors():
    an = Animation([0.0, 1.0])
    with pytest.raises(ValueError):
        an.add_track("b", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        an.add_vectors("v", np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), attach="missing")
    with pytest.raises(ValueError):
        Animation([0.0, 0.0])


def test_scene_extras_and_web_viewer(tmp_path):
    from pinneapple_tools.visualization.studio.web import web_viewer
    sc = _scene()
    ex = sc.extras()
    assert "animation" in ex and ex["animation"]["tracks"][0]["surfaces"] == [1]
    assert ex["animation"]["fields"][0]["index"] == 1
    assert "animation" not in Scene(sc.surfaces).extras()
    web_viewer(sc, str(tmp_path))
    meta = json.load(open(tmp_path / "scene.json"))
    assert len(meta["animation"]["times"]) == 6 and meta["animation"]["vectors"][0]["attach"] == "body"
    js = open(tmp_path / "viewer.js").read()
    assert "setFrame" in js and "studio-core.js" in js


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def _chromium_ok():
    if not _has("playwright"):
        return False
    from pinneapple_tools.visualization.studio.web import _chromium_path
    return _chromium_path() is not None or bool(os.environ.get("PINNEAPPLE_PLAYWRIGHT_BUNDLED"))


@pytest.mark.skipif(not _chromium_ok(), reason="playwright / Chromium not available")
def test_viewer_screenshot_moves(tmp_path):
    from PIL import Image

    from pinneapple_tools.visualization.studio.web import screenshot, web_viewer
    d = tmp_path / "v"
    web_viewer(_scene(), str(d))
    a = screenshot(str(d), str(tmp_path / "f0.png"), frame=0, size=(480, 300))
    b = screenshot(str(d), str(tmp_path / "f5.png"), frame=5, size=(480, 300))
    A, Bm = (np.asarray(Image.open(p).convert("L"), float) for p in (a, b))
    assert A.shape == (300, 480) and np.abs(A - Bm).mean() > 0.05     # the body (and the HUD) moved


@pytest.mark.skipif(not _has("bpy"), reason="bpy (Blender) not installed")
def test_blender_render_animation(tmp_path):
    from pinneapple_tools.visualization.studio.blender import render_animation
    sc = _scene()
    r = render_animation(sc, str(tmp_path), frames=[0, 3], size=(64, 36), samples=1, gif=True)
    assert len(r["frames"]) == 2 and all(os.path.getsize(p) > 0 for p in r["frames"])
    assert r["mp4"] and os.path.getsize(r["mp4"]) > 0
    assert r["gif"] and os.path.getsize(r["gif"]) > 0
