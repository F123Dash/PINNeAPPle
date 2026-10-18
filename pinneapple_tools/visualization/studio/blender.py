"""Blender Cycles renders of any :class:`~.scene.Scene` (``pip install bpy``, or run inside Blender).

    from pinneapple_tools.visualization.studio import Scene, render
    render(scene, "cp.jpg", field="cp")                          # skin coloured by a field, jet scale, colour bar
    render(scene, "flow.jpg", lines=True)                        # streamlines coloured by their values
    render(scene, "slice.jpg", slice="wake", view="back")        # a slice plane (NaN = transparent)
    render(scene, "hero.jpg", background="sky")                  # the materials, under a physical sky

``view``: "iso" (default), "iso_back", "iso_low", "front", "back", "side", "side_left", "top", "bottom", or a
direction (dx, dy, dz) from the centre towards the camera, in scene axes. Names assume x is the object's length
(the flow direction for a CFD scene) and z is up: "front" looks at the x-min end. The colour bar title comes from
``scene.labels[field]`` unless ``title`` is given.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .colormap import JET, colorbar, jet

VIEWS = {   # direction from the centre to the camera; x is the object's length / the flow direction, z up
    "iso": (-1.0, -1.1, 0.75), "iso_back": (1.0, -1.1, 0.75), "iso_low": (-1.0, -1.1, -0.4),
    "front": (-1.0, 0.0, 0.12), "back": (1.0, 0.0, 0.12), "side": (0.0, -1.0, 0.12), "side_left": (0.0, 1.0, 0.12),
    "top": (0.001, 0.0, 1.0), "bottom": (0.001, 0.0, -1.0),
}


def _lin(c):
    return tuple(v ** 2.2 for v in c)                     # Blender colour sockets are linear


def _to_blender(P: np.ndarray, axes: str) -> np.ndarray:
    """Scene axes -> Blender (z up). The glTF importer undoes AXES[axes]; this matches it for lines and slices."""
    P = np.asarray(P, float)
    if axes == "aircraft":
        return np.c_[P[:, 1], -P[:, 0], P[:, 2]]
    return P


def render(scene, out: str, *, field: Optional[str] = None, field_range: Optional[Tuple[float, float]] = None,
           lines: Union[bool, str] = False, line_range: Optional[Tuple[float, float]] = None,
           slice: Optional[str] = None, slice_range: Optional[Tuple[float, float]] = None,
           view: Union[str, Sequence[float]] = "iso", distance: float = 1.0, lens: float = 50,
           background: str = "studio", samples: int = 96, size: Tuple[int, int] = (1600, 900),
           title: Optional[str] = None, colorbar_on: bool = True, lo_txt: str = "", hi_txt: str = "",
           sun_elevation: float = 40.0, ghost: Optional[bool] = None) -> str:
    """Render ``scene`` (a Scene or a .glb path) to ``out`` (.png or .jpg). Returns ``out``.

    field: colour the surfaces by this per-vertex field (surfaces without it are grey). lines: True (all) or the
    name of a Lines set, coloured by their values. slice: name of a Slice. background: "studio" (light gradient,
    true colours), "sky" (Nishita sky, filmic) or a lunar preset of :mod:`.lighting` ("lunar_south_pole",
    "lunar_low_sun", "lunar_day": black sky, one Sun lamp with the Sun's angular size). ghost: draw the surfaces translucent (default: when a slice is
    shown)."""
    import bpy
    from mathutils import Vector

    if isinstance(scene, str):
        glb, axes, sc = scene, "z_up", None
    else:
        sc = scene
        axes = sc.axes
        glb = os.path.join(tempfile.mkdtemp(), "scene.glb")
        sc.save(glb)
    studio = background == "studio"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    bs.cycles.device = "CPU"
    bs.cycles.samples = samples
    bs.cycles.use_denoising = True
    bs.render.resolution_x, bs.render.resolution_y = size
    names = [i.identifier for i in bs.view_settings.bl_rna.properties["view_transform"].enum_items]
    bs.view_settings.view_transform = "Standard" if studio else ("AgX" if "AgX" in names else "Filmic")
    bs.view_settings.look = "None"
    bs.view_settings.exposure = 0.0 if (studio and (field or slice)) else (-0.5 if studio else -0.3)

    bpy.ops.import_scene.gltf(filepath=glb)
    objs = [o for o in bs.objects if o.type == "MESH"]
    mins = Vector((min(min((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    maxs = Vector((max(max((o.matrix_world @ Vector(c))[i] for c in o.bound_box) for o in objs) for i in range(3)))
    centre, size_l = (mins + maxs) / 2, (maxs - mins).length

    def ramp_material(name: str, attr: str, lo: float, hi: float):
        return _ramp_material(bpy, name, attr, lo, hi)

    def plain(name, rgb, alpha=1.0):
        return _plain(bpy, name, rgb, alpha)

    bar = None
    if field:
        attr = "_" + field.upper().lstrip("_")
        lo, hi = field_range or (sc.field_range(field) if sc else (0.0, 1.0))
        fm, grey = ramp_material(field, attr, lo, hi), plain("grey", (0.35, 0.37, 0.4))
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(fm if attr in o.data.attributes else grey)
        bar = (title or (sc.labels.get(field, field) if sc else field), lo, hi)

    show_ghost = ghost if ghost is not None else bool(slice)
    if show_ghost and not field:
        gm = plain("ghost", (0.72, 0.75, 0.78), 0.35)
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(gm)

    if lines and sc is not None:
        sets = [L for L in sc.lines if lines is True or L.name == lines]
        allv = np.concatenate([np.concatenate(L.values) for L in sets if L.values]) if any(L.values for L in sets) else None
        lo, hi = line_range or ((float(np.nanpercentile(allv, 2)), float(np.nanpercentile(allv, 98))) if allv is not None else (0, 1))
        nb = 24
        mats = []
        for j in range(nb):
            m = bpy.data.materials.new(f"line{j}")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            c = _lin(jet((j + 0.5) / nb)) if allv is not None else (0.05 ** 2.2, 0.75 ** 2.2, 1.0)
            b.inputs["Base Color"].default_value = (*c, 1)
            b.inputs["Emission Color"].default_value = (*c, 1)
            b.inputs["Emission Strength"].default_value = 1.6
            mats.append(m)
        k = 0
        for L in sets:
            for li, P0 in enumerate(L.points):
                P = _to_blender(P0, axes)
                S = np.asarray(L.values[li], float) if L.values else None
                keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9 * size_l]
                P = P[keep]
                S = S[keep] if S is not None else None
                if len(P) < 2:
                    continue
                cu = bpy.data.curves.new(f"line{k}", "CURVE")
                cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.0011 * size_l, 2
                for m in mats:
                    cu.materials.append(m)
                step = 3 if S is not None else len(P)
                for i0 in range(0, len(P) - 1, step):
                    seg = P[i0:i0 + step + 1]
                    sp = cu.splines.new("POLY")
                    sp.points.add(len(seg) - 1)
                    for i, p in enumerate(seg):
                        sp.points[i].co = (p[0], p[1], p[2], 1)
                    if S is not None:
                        t = (np.mean(S[i0:i0 + step + 1]) - lo) / ((hi - lo) or 1)
                        sp.material_index = int(min(nb - 1, max(0, t * nb)))
                ob = bpy.data.objects.new(f"line{k}", cu)
                bs.collection.objects.link(ob)
                k += 1
        if bar is None and allv is not None:
            lab = next((L.label for L in sets if L.label), "")
            bar = (title or lab or "value along the lines", lo, hi)

    if slice and sc is not None:
        s = next(x for x in sc.slices if x.name == slice)
        lo, hi = slice_range or (float(np.nanpercentile(s.grid, 2)), float(np.nanpercentile(s.grid, 98)))
        from PIL import Image
        g = np.asarray(s.grid, float)
        rgba = np.zeros(g.shape + (4,), np.uint8)
        ok = np.isfinite(g)
        t = np.clip((np.where(ok, g, lo) - lo) / ((hi - lo) or 1), 0, 1)
        lut = np.array([jet(v) for v in np.linspace(0, 1, 256)])
        rgba[..., :3] = (lut[(t * 255).astype(int)] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(ok, 240, 0)
        png = os.path.join(tempfile.mkdtemp(), "slice.png")
        Image.fromarray(rgba[::-1]).save(png)                    # row 0 of the grid is at the bottom (s = 0 side)
        C = _to_blender(np.array([s.origin, s.origin + s.u, s.origin + s.u + s.v, s.origin + s.v]), axes)
        me = bpy.data.meshes.new("slice")
        me.from_pydata([tuple(p) for p in C], [], [(0, 1, 2, 3)])
        uv = me.uv_layers.new()
        for li, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv.data[li].uv = (u, v)
        m = bpy.data.materials.new("slice")
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(png)
        tex.interpolation = "Closest" if min(g.shape) < 60 else "Linear"
        em = nt.nodes.new("ShaderNodeEmission")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        outn = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
        nt.links.new(em.outputs["Emission"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], outn.inputs["Surface"])
        me.materials.append(m)
        ob = bpy.data.objects.new("slice", me)
        bs.collection.objects.link(ob)
        if bar is None:
            bar = (title or s.label or slice, lo, hi)

    if background.startswith("lunar"):                  # black sky, one hard Sun (lighting.PRESETS)
        from .lighting import blender_world
        blender_world(background, bpy)
    else:
        _world(bpy, bs, studio, sun_elevation)

    d = _to_blender(np.asarray(VIEWS[view] if isinstance(view, str) else view, float)[None], axes)[0]
    d = d / np.linalg.norm(d)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam = bpy.data.objects.new("cam", cam_data)
    bs.collection.objects.link(cam)
    cam_data.sensor_fit = "AUTO"                         # the lens angle spans the longer side of the image
    t_long = math.tan(0.5 * cam_data.angle)
    t_short = t_long * min(size) / max(size)
    tx, ty = (t_long, t_short) if size[0] >= size[1] else (t_short, t_long)
    # distance at which every corner of the bounding box is inside the frame
    fwd = -d
    upw = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, upw)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c0 = np.array(centre)
    pts = []                                             # the visible geometry, not its bounding box
    for o in objs:
        mw = np.array(o.matrix_world)
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        if len(co) > 4000:
            co = co[:: len(co) // 4000]
        pts.append(co @ mw[:3, :3].T + mw[:3, 3])
    if slice and sc is not None:                         # frame the whole slice plane, not only the body
        sl = next(x for x in sc.slices if x.name == slice)
        pts.append(_to_blender(np.array([sl.origin, sl.origin + sl.u, sl.origin + sl.u + sl.v, sl.origin + sl.v]), axes))
    k = np.concatenate(pts) - c0
    # aim at the middle of the projected extent, then back off until every vertex is in frame (perspective)
    shift = right * 0.5 * (np.max(k @ right) + np.min(k @ right)) + up * 0.5 * (np.max(k @ up) + np.min(k @ up))
    centre = centre + Vector(tuple(shift))
    k = k - shift
    lat = np.maximum(np.abs(k @ right) / tx, np.abs(k @ up) / ty)
    fit = float(np.max(k @ d + lat)) * 1.15 * distance
    cam.location = centre + Vector(tuple(d * fit))
    cam_data.clip_start, cam_data.clip_end = fit * 1e-3, fit * 1e3     # small models: the default 0.1 m cuts them
    look = centre - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    bs.camera = cam

    if out.lower().endswith((".jpg", ".jpeg")):
        bs.render.image_settings.file_format = "JPEG"
        bs.render.image_settings.quality = 92
    bs.render.filepath = os.path.abspath(out)
    bpy.ops.render.render(write_still=True)
    if bar and colorbar_on:
        colorbar(out, bar[0], bar[1], bar[2], None, lo_txt, hi_txt)
    return out


def _ramp_material(bpy, name: str, attr: str, lo: float, hi: float):
    """Principled material coloured on the jet scale by the mesh attribute ``attr`` mapped from lo..hi."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = attr
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = lo, hi
    rp = nt.nodes.new("ShaderNodeValToRGB")
    el = rp.color_ramp.elements
    el[0].position, el[0].color = 0.0, (*_lin(jet(0.0)), 1)
    el[1].position, el[1].color = 1.0, (*_lin(jet(1.0)), 1)
    for t, _ in JET[1:-1]:
        e = el.new(t)
        e.color = (*_lin(jet(t)), 1)
    nt.links.new(a.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], rp.inputs["Fac"])
    nt.links.new(rp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(rp.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.35               # readable in shadow
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def _plain(bpy, name, rgb, alpha=1.0):
    """Principled material of one linear colour (translucent when alpha < 1)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    return m


def _world(bpy, bs, studio, sun_elevation):
    """Studio gradient or Nishita sky world with a Sun lamp."""
    import math
    world = bpy.data.worlds.new("world")
    bs.world = world
    world.use_nodes = True
    wn = world.node_tree
    if studio:
        grad, tc, mp = wn.nodes.new("ShaderNodeTexGradient"), wn.nodes.new("ShaderNodeTexCoord"), wn.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, math.radians(-90), 0)
        cr = wn.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (0.62, 0.66, 0.72, 1)
        cr.color_ramp.elements[1].color = (0.95, 0.96, 0.98, 1)
        wn.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        wn.links.new(mp.outputs["Vector"], grad.inputs["Vector"])
        wn.links.new(grad.outputs["Fac"], cr.inputs["Fac"])
        wn.links.new(cr.outputs["Color"], wn.nodes["Background"].inputs["Color"])
        wn.nodes["Background"].inputs["Strength"].default_value = 0.9
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy, sun.angle = 2.2, math.radians(8)
        so = bpy.data.objects.new("sun", sun)
        so.rotation_euler = (math.radians(35), math.radians(-20), math.radians(30))
        bs.collection.objects.link(so)
    else:
        sky = wn.nodes.new("ShaderNodeTexSky")
        kinds = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
        sky.sky_type = "NISHITA" if "NISHITA" in kinds else "MULTIPLE_SCATTERING"        # renamed in Blender 5
        sky.sun_elevation, sky.sun_rotation = math.radians(sun_elevation), math.radians(215)
        sky.altitude = 2000
        bg = wn.nodes["Background"]
        bg.inputs["Strength"].default_value = 0.12
        wn.links.new(sky.outputs["Color"], bg.inputs["Color"])


# ---------------------------------------------------------------------- animations
_GLTF_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])     # glTF y up -> Blender z up


def blender_axes(axes: str) -> np.ndarray:
    """Rotation matrix scene axes -> Blender coordinates (what the glTF export + import does to the surfaces)."""
    from .scene import AXES
    return _GLTF_TO_BLENDER @ AXES[axes]


def _arrow_meshes(bpy, n: int = 12):
    """Unit cylinder (radius 1, z from 0 to 1) and unit cone (base radius 1 at z = 0, tip at z = 1)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    cyl_v = [(x, y, 0.0) for x, y in ring] + [(x, y, 1.0) for x, y in ring] + [(0, 0, 0), (0, 0, 1)]
    cyl_f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    cyl_f += [(2 * n, (i + 1) % n, i) for i in range(n)] + [(2 * n + 1, n + i, n + (i + 1) % n) for i in range(n)]
    cone_v = [(x, y, 0.0) for x, y in ring] + [(0, 0, 1), (0, 0, 0)]
    cone_f = [(i, (i + 1) % n, n) for i in range(n)] + [(n + 1, (i + 1) % n, i) for i in range(n)]
    out = []
    for name, V, F in (("arrow_shaft", cyl_v, cyl_f), ("arrow_head", cone_v, cone_f)):
        me = bpy.data.meshes.new(name)
        me.from_pydata(V, [], F)
        me.update()
        out.append(me)
    return out


def _hud_overlay(path: str, lines: Sequence[str]) -> None:
    """Burn the head-up lines (first one bold) into the top-left corner of the image at ``path``."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    W, H = im.size
    fs = max(8, int(round(H * 0.032)))
    try:
        f = ImageFont.truetype("DejaVuSansMono.ttf", fs)
        fb = ImageFont.truetype("DejaVuSansMono-Bold.ttf", fs)
    except OSError:
        f = fb = ImageFont.load_default()
    d = ImageDraw.Draw(im, "RGBA")
    pad, lh = max(3, fs // 2), int(fs * 1.35)
    w = max(d.textlength(t, font=fb if i == 0 else f) for i, t in enumerate(lines))
    d.rounded_rectangle([pad, pad, pad * 3 + w, pad * 3 + lh * len(lines) - (lh - fs)], radius=pad,
                        fill=(15, 23, 42, 190))
    for i, t in enumerate(lines):
        d.text((pad * 2, pad * 2 + i * lh), t, fill=(248, 250, 252), font=fb if i == 0 else f)
    im.save(path)


def _encode(pngs: Sequence[str], out: str, fps: float, gif: bool = False, gif_fps: float = 12,
            gif_width: int = 480) -> str:
    """MP4 (H.264, yuv420p, even size) or GIF (palette) of a PNG sequence with the ffmpeg binary; without it, the
    MP4 goes through Blender's encoder (:func:`.particles.encode_mp4`) and the GIF through PIL."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if exe is None:
        if gif:
            from PIL import Image
            ims = [Image.open(p).convert("RGB") for p in pngs]
            ims = [im.resize((gif_width, int(im.height * gif_width / im.width))) if im.width > gif_width else im
                   for im in ims]
            ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
            return out
        from .particles import encode_mp4
        return encode_mp4(list(pngs), out, fps=int(round(fps)))
    import tempfile as _tf
    seq = _tf.mkdtemp()                                         # numbered links: ffmpeg's image2 pattern
    for i, p in enumerate(pngs):
        os.symlink(os.path.abspath(p), os.path.join(seq, f"f_{i:06d}.png"))
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", os.path.join(seq, "f_%06d.png")]
    if gif:
        vf = (f"fps={min(fps, gif_fps)},scale='min({gif_width},iw)':-2:flags=lanczos,split[a][b];"
              "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        cmd += ["-vf", vf, "-loop", "0", out]
    else:
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", f"{fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(seq, ignore_errors=True)
    return out


def render_animation(scene, out_dir: str, *, frames: int | Sequence[int] | None = None,
                     size: tuple[int, int] = (960, 540), samples: int = 16, camera: str = "chase",
                     camera_offset: Sequence[float] | None = None, lens: float = 35, field: str | None = None,
                     hud: bool = True, fps: float = 24, gif: bool = False, mp4: bool = True,
                     sun_elevation: float = 35.0, ground: bool | float | None = None,
                     background: str = "sky", vectors: bool = True, trails: bool = True,
                     colorbar_on: bool = True, name: str = "animation", log=None) -> dict:
    """Render ``scene.animation`` (an :class:`~.animation.Animation`) frame by frame with Cycles (CPU, denoised),
    burn the HUD into each PNG and encode an MP4 (and a GIF). Returns ``{"frames": [png, ...], "mp4": path | None,
    "gif": path | None}``.

    frames: None (every frame), an int n (the animation resampled to n evenly spaced frames) or frame indices.
    camera: "chase" (behind the tracked body along its direction of travel, or at ``camera_offset`` in the body
    frame), "follow" (at ``camera_offset`` in scene axes from the body, default from the 3/4 view), "side"
    (alongside the path, tracking the body), "tower" (fixed at ``camera_offset`` in scene axes, default beside the
    middle of the path, looking at the body) or "fixed" (the whole scene and path in the 3/4 view).
    field: colour the surfaces by this static per-vertex field; animated surface fields are always shown.
    ground: True for a grey plane under the scene and path, or a height (scene up axis) for its level.
    background: "sky" (physical sky) or "studio" (light gradient)."""
    import bpy
    from mathutils import Quaternion, Vector

    from .scene import Scene, Surface
    anim = scene.animation
    if anim is None:
        raise ValueError("scene.animation is None: build an Animation and set scene.animation first")
    if isinstance(frames, int):
        anim = anim.resample(frames)
        frames = None
    ks = list(range(anim.n_frames)) if frames is None else [int(k) for k in frames]
    os.makedirs(out_dir, exist_ok=True)
    B = blender_axes(scene.axes)
    toB = lambda P: np.asarray(P, float) @ B.T                         # noqa: E731
    qB = lambda q: Quaternion((q[0], *(B @ np.asarray(q[1:], float))))  # noqa: E731

    # surfaces with unique node names so the imported objects map back to them
    tagged = [Surface(f"s{i:04d}", s.vertices, s.faces, s.material, s.fields, s.group) for i, s in enumerate(scene.surfaces)]
    glb = os.path.join(tempfile.mkdtemp(), "scene.glb")
    Scene(tagged, axes=scene.axes).save(glb)
    studio = background == "studio"

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    bs.cycles.device = "CPU"
    bs.cycles.samples = max(1, int(samples))
    bs.cycles.use_denoising = True
    try:
        bs.cycles.denoiser = "OPENIMAGEDENOISE"
    except (TypeError, AttributeError):
        pass
    if hasattr(bs.render, "use_persistent_data"):
        bs.render.use_persistent_data = True
    bs.render.resolution_x, bs.render.resolution_y = size
    bs.render.resolution_percentage = 100
    names = [i.identifier for i in bs.view_settings.bl_rna.properties["view_transform"].enum_items]
    bs.view_settings.view_transform = "Standard" if studio else ("AgX" if "AgX" in names else "Filmic")
    bs.view_settings.exposure = -0.3 if not studio else -0.2
    bs.render.image_settings.file_format = "PNG"

    bpy.ops.import_scene.gltf(filepath=glb)
    objs = {}
    for o in list(bs.objects):
        if o.type == "MESH" and o.name[:1] == "s" and o.name[1:5].isdigit():
            objs[int(o.name[1:5])] = o

    # extent of the static scene and of the paths (for the ground, the fixed camera and the clip planes)
    static = [toB(s.vertices) for i, s in enumerate(scene.surfaces) if s.group not in anim.tracks]
    lo_p, hi_p = anim.bounds()
    pts = np.concatenate(static + [toB(np.array([lo_p, hi_p]))]) if (static or anim.tracks or anim.trails) else np.zeros((1, 3))
    s_lo, s_hi = pts.min(0), pts.max(0)
    extent = float(np.linalg.norm(s_hi - s_lo)) or 1.0

    # tracks: the surfaces of each group under an Empty
    empties, radius = {}, {}
    for g in anim.tracks:
        e = bpy.data.objects.new(f"track_{g}", None)
        e.rotation_mode = "QUATERNION"
        bs.collection.objects.link(e)
        empties[g] = e
        V = [toB(s.vertices) for s in scene.surfaces if s.group == g]
        V = np.concatenate(V) if V else np.zeros((1, 3))
        radius[g] = max(1e-6, 0.5 * float(np.linalg.norm(V.max(0) - V.min(0))))
        for i, s in enumerate(scene.surfaces):
            if s.group == g and i in objs:
                objs[i].parent = e
    fg = anim.follow_group if anim.follow_group in anim.tracks else next(iter(anim.tracks), None)
    body_r = radius.get(fg, extent / 20)

    # colours: a static field, then the animated per-vertex fields
    bar = None
    if field:
        attr = "_" + field.upper().lstrip("_")
        lo, hi = scene.field_range(field)
        fm, grey = _ramp_material(bpy, field, attr, lo, hi), _plain(bpy, "grey", (0.35, 0.37, 0.4))
        for o in objs.values():
            o.data.materials.clear()
            o.data.materials.append(fm if attr in o.data.attributes else grey)
        bar = (scene.labels.get(field, field), lo, hi)
    afields = []
    for sname, sf in anim.surface_fields.items():
        idx = next((i for i, s in enumerate(scene.surfaces) if s.name == sname), None)
        if idx is None or idx not in objs:
            continue
        o = objs[idx]
        me = o.data
        nv = len(me.vertices)
        if nv == len(scene.surfaces[idx].vertices):
            order = None
        else:                                                   # the importer split or merged vertices: nearest
            from scipy.spatial import cKDTree
            co = np.empty(nv * 3)
            me.vertices.foreach_get("co", co)
            order = cKDTree(toB(scene.surfaces[idx].vertices)).query(co.reshape(-1, 3))[1]
        at = me.attributes.new("anim_field", "FLOAT", "POINT")
        me.materials.clear()
        me.materials.append(_ramp_material(bpy, f"anim_{sname}", "anim_field", *sf["range"]))
        afields.append((sf, at, order, me))
        if bar is None:
            bar = (sf["label"] or sname, *sf["range"])

    # arrows: one shaft and one head object per arrow, parented to the track when attached
    arrows = []
    if vectors and anim.vectors:
        shaft_me, head_me = _arrow_meshes(bpy)
        placeholder = _plain(bpy, "arrow", (0.8, 0.3, 0.05))
        shaft_me.materials.append(placeholder)
        head_me.materials.append(placeholder)
        for vi, vs in enumerate(anim.vectors):
            L = float(np.nanmax(np.linalg.norm(vs["vectors"], axis=2))) or 1.0
            col = vs["color"].lstrip("#")
            base = tuple(int(col[j:j + 2], 16) / 255 for j in (0, 2, 4))
            for i in range(vs["origins"].shape[1]):
                m = _plain(bpy, f"arrow{vi}_{i}", _lin(base))
                m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.4
                pair = []
                for kind, me in (("shaft", shaft_me), ("head", head_me)):
                    ob = bpy.data.objects.new(f"arrow{vi}_{i}_{kind}", me)
                    ob.rotation_mode = "QUATERNION"
                    if vs["attach"] in empties:
                        ob.parent = empties[vs["attach"]]
                    bs.collection.objects.link(ob)
                    pair.append(ob)
                arrows.append((vs, i, pair, m, L))
            if bar is None and vs["values"] is not None and vs["range"] is not None:
                bar = (vs["label"] or vs["name"], *vs["range"])
    # the arrow meshes are shared: each object gets its own material through an object-linked slot
    for _, _, pair, m, _ in arrows:
        for ob in pair:
            ob.material_slots[0].link = "OBJECT"
            ob.material_slots[0].material = m

    trail_objs = []
    if trails:
        for tr in anim.trails:
            P = toB(tr["points"])
            cu = bpy.data.curves.new(tr["name"], "CURVE")
            cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.04 * body_r, 2
            sp = cu.splines.new("POLY")
            sp.points.add(len(P) - 1)
            for i, p in enumerate(P):
                sp.points[i].co = (p[0], p[1], p[2], 1)
            cu.bevel_factor_mapping_end = "SEGMENTS"
            col = tr["color"].lstrip("#")
            c = _lin(tuple(int(col[j:j + 2], 16) / 255 for j in (0, 2, 4)))
            m = _plain(bpy, f"trail_{tr['name']}", c)
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Emission Color"].default_value = (*c, 1)
            b.inputs["Emission Strength"].default_value = 1.5
            cu.materials.append(m)
            ob = bpy.data.objects.new(tr["name"], cu)
            bs.collection.objects.link(ob)
            trail_objs.append((ob, len(P)))

    if ground is not None and ground is not False:
        # Blender z is the scene's up axis; the automatic level sits a little below everything (no z-fighting with
        # a runway or floor surface of the scene)
        z = float(s_lo[2]) - 1e-3 * extent if ground is True else float(ground)
        c = 0.5 * (s_lo + s_hi)
        h = 1.5 * extent
        me = bpy.data.meshes.new("ground")
        me.from_pydata([(c[0] - h, c[1] - h, z), (c[0] + h, c[1] - h, z), (c[0] + h, c[1] + h, z),
                        (c[0] - h, c[1] + h, z)], [], [(0, 1, 2, 3)])
        gm = _plain(bpy, "ground", (0.30, 0.31, 0.30))
        gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
        me.materials.append(gm)
        bs.collection.objects.link(bpy.data.objects.new("ground", me))

    _world(bpy, bs, studio, sun_elevation)
    if not studio:                                             # a Sun lamp matching the sky, for crisp shadows
        sun = bpy.data.lights.new("sun_lamp", "SUN")
        sun.energy, sun.angle = 3.0, math.radians(1.5)
        so = bpy.data.objects.new("sun_lamp", sun)
        so.rotation_euler = (math.radians(90 - sun_elevation), 0, math.radians(215 + 90))
        bs.collection.objects.link(so)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam_data.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("cam", cam_data)
    bs.collection.objects.link(cam)
    bs.camera = cam
    half = 0.5 * cam_data.angle * min(size) / max(size)       # half angle across the shorter image side
    chase_d = body_r / math.tan(half) * 1.7

    def body_at(k):
        if fg is None:
            return 0.5 * (s_lo + s_hi), np.eye(3)
        M = anim.pose(fg, k)
        V = [s.vertices for s in scene.surfaces if s.group == fg]
        c_body = 0.5 * (np.concatenate(V).min(0) + np.concatenate(V).max(0)) if V else np.zeros(3)
        return B @ (M[:3, :3] @ c_body + M[:3, 3]), B @ M[:3, :3] @ B.T

    def travel(k, win=3):
        P = toB(anim.tracks[fg]["positions"]) if fg else np.zeros((anim.n_frames, 3))
        a, b = max(0, k - win), min(anim.n_frames - 1, k + win)
        v = P[b] - P[a]
        v[2] = 0.0
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 * extent else np.array([1.0, 0.0, 0.0])

    path = toB(anim.tracks[fg]["positions"]) if fg else pts
    p_dir = path[-1] - path[0]
    p_dir[2] = 0
    p_dir = p_dir / np.linalg.norm(p_dir) if np.linalg.norm(p_dir) > 1e-9 * extent else np.array([1.0, 0.0, 0.0])
    side_n = np.cross([0.0, 0.0, 1.0], p_dir)
    tower = (B @ np.asarray(camera_offset, float) if camera_offset is not None and camera == "tower"
             else 0.5 * (path.min(0) + path.max(0)) - side_n * max(4 * chase_d, 0.25 * extent) + np.array([0, 0, chase_d * 0.3]))
    iso = _to_blender(np.asarray(VIEWS["iso"], float)[None], "z_up")[0]
    iso /= np.linalg.norm(iso)

    def place_camera(k):
        target, Rb = body_at(k)
        if camera == "chase":
            off = (Rb @ (B @ np.asarray(camera_offset, float)) if camera_offset is not None
                   else -travel(k) * chase_d * 0.92 + np.array([0, 0, chase_d * 0.35]))
            loc = target + off
        elif camera == "follow":
            loc = target + (B @ np.asarray(camera_offset, float) if camera_offset is not None else iso * chase_d)
        elif camera == "side":
            loc = target - side_n * chase_d + np.array([0, 0, 0.15 * chase_d])
        elif camera == "tower":
            loc = tower
        elif camera == "fixed":
            target = 0.5 * (s_lo + s_hi)
            loc = target + iso * 0.5 * extent / math.tan(half) * 0.8
        else:
            raise ValueError(f"camera={camera!r}: chase, follow, side, tower or fixed")
        cam.location = Vector(tuple(loc))
        look = Vector(tuple(target)) - cam.location
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        d = max(look.length, 1e-6)
        cam_data.clip_start = max(d * 1e-3, 1e-5)
        cam_data.clip_end = max(d * 1e3, 3 * extent + d)

    pngs = []
    for j, k in enumerate(ks):
        for g, e in empties.items():
            tr = anim.tracks[g]
            e.location = Vector(tuple(B @ tr["positions"][k]))
            e.rotation_quaternion = qB(tr["quaternions"][k])
        for vs, i, (shaft, head), m, L in arrows:
            o, v = vs["origins"][k, i], vs["vectors"][k, i]
            o, v = B @ o, B @ v                                  # attached arrows: body coordinates under the Empty
            n = float(np.linalg.norm(v))
            show = n > 1e-9 * L
            for ob in (shaft, head):
                ob.hide_render = not show
            if not show:
                continue
            u = v / n
            q = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(tuple(u)))
            h, r = min(0.22 * L, 0.45 * n), min(0.022 * L, 0.25 * n)
            shaft.location, shaft.rotation_quaternion, shaft.scale = Vector(tuple(o)), q, (r, r, n - h)
            head.location, head.rotation_quaternion, head.scale = Vector(tuple(o + u * (n - h))), q, (2.6 * r, 2.6 * r, h)
            if vs["values"] is not None and vs["range"] is not None:
                lo, hi = vs["range"]
                c = _lin(jet((vs["values"][k, i] - lo) / ((hi - lo) or 1)))
                b = m.node_tree.nodes["Principled BSDF"]
                b.inputs["Base Color"].default_value = (*c, 1)
                b.inputs["Emission Color"].default_value = (*c, 1)
        for ob, n in trail_objs:
            ob.data.bevel_factor_end = min(1.0, k / max(1, n - 1)) if k > 0 else 1e-4
        for sf, at, order, me in afields:
            vals = np.asarray(sf["values"][k], float)
            vals = np.where(np.isfinite(vals), vals, sf["range"][0])
            at.data.foreach_set("value", (vals if order is None else vals[order]).astype(np.float32))
            me.update()
        bpy.context.view_layer.update()
        place_camera(k)
        path_png = os.path.join(out_dir, f"frame_{j:05d}.png")
        bs.render.filepath = os.path.abspath(path_png)
        bpy.ops.render.render(write_still=True)
        if bar and colorbar_on:
            colorbar(path_png, bar[0], bar[1], bar[2])
        if hud:
            _hud_overlay(path_png, anim.hud_lines(k))
        pngs.append(path_png)
        if log and (j + 1) % 10 == 0:
            log(f"rendered {j + 1}/{len(ks)} frames")
    out = {"frames": pngs, "mp4": None, "gif": None}
    if mp4 and pngs:
        out["mp4"] = _encode(pngs, os.path.join(out_dir, f"{name}.mp4"), fps)
    if gif and pngs:
        out["gif"] = _encode(pngs, os.path.join(out_dir, f"{name}.gif"), fps, gif=True)
    return out
