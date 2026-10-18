"""Independent cross-check of the machine FEA with gmsh (mesh) and GetDP (solver).

gmsh (https://gmsh.info) and GetDP (https://getdp.info) are GPL programs used here as external executables only
(``gmsh`` and ``getdp`` on the path; no code of theirs is copied or linked). For one rotor angle and one set of
phase currents, :func:`solve` writes

* a gmsh geometry (``.geo``, built-in kernel) of the full cross-section from the same polygons as the mesher of
  :mod:`.mesher` (:func:`write_geo`): every region is a plane surface bounded by shared lines, iron regions with
  the pole and slot holes as inner loops; the rotor is turned by ``theta``, so each angle has its own unstructured
  mesh (no moving band);
* a GetDP problem (``.pro``, :func:`write_pro`): A_z magnetostatics in ``H(curl)`` with perpendicular edge
  (nodal) functions, nonlinear iron from the B-H table (Akima interpolation of ``nu(b^2)``, Newton-Raphson with the
  ``JacNL`` tangent term), linear magnets with remanence, coil current densities, ``A = 0`` on the shaft and outer
  circles; post-processing in GetDP of the Arkkio torque over the air gap and of the three phase flux linkages,

runs both programs and returns their numbers. Nothing of this package's own mesher, solver or post-processing is
used for the result, so the agreement with :class:`~.pmsm.IPMMachine` measures discretisation and implementation
differences between two independent codes on identical geometry and material data.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .geometry import MachineGeometry, polygon_area

__all__ = ["available", "write_geo", "write_pro", "solve", "GmshGetDPError"]

TWO_PI = 2.0 * math.pi
MU0 = 4e-7 * math.pi


class GmshGetDPError(RuntimeError):
    pass


def available() -> bool:
    return shutil.which("gmsh") is not None and shutil.which("getdp") is not None


def _circle(r: float, n: int, extra: list[float] | None = None) -> np.ndarray:
    ang = np.sort(np.unique(np.round(np.concatenate([TWO_PI * np.arange(n) / n, extra or []]) % TWO_PI, 14)))
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


class _PSLG:
    """Shared vertices and segments of closed polygons; regions as loops of signed segments."""

    def __init__(self, tol: float = 1e-10):
        self.pts: list[tuple[float, float]] = []
        self.index: dict = {}
        self.tol = tol
        self.lc: list[float] = []

    def vid(self, p, lc):
        key = (round(p[0] / self.tol), round(p[1] / self.tol))
        if key not in self.index:
            self.index[key] = len(self.pts)
            self.pts.append((float(p[0]), float(p[1])))
            self.lc.append(float(lc))
        return self.index[key]

    def loop(self, xy, lcf):
        ids = [self.vid(p, lcf(p)) for p in xy]
        return [i for k, i in enumerate(ids) if i != ids[k - 1]]


def _edges(loop):
    return [(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))]


def _boundary_loops(loops: list[list[int]], pts) -> list[list[int]]:
    """Closed loops of the edges that appear an odd number of times among ``loops`` (outer loop first)."""
    count: dict = {}
    for lp in loops:
        for a, b in _edges(lp):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    edges = [k for k, c in count.items() if c % 2 == 1]
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used = set()
    out = []
    P = np.asarray(pts)
    for a0, b0 in edges:
        if (a0, b0) in used:
            continue
        lp = [a0]
        prev, cur = a0, b0
        used.add((min(a0, b0), max(a0, b0)))
        while cur != a0:
            lp.append(cur)
            nxt = [v for v in adj[cur] if (min(cur, v), max(cur, v)) not in used]
            if not nxt:
                raise GmshGetDPError("open boundary while tracing a region")
            if len(nxt) > 1:  # pinch point: continue along the edge making the sharpest left turn
                d0 = P[cur] - P[prev]
                def turn(v, d0=d0, cur=cur):
                    e = P[v] - P[cur]
                    return -math.atan2(d0[0] * e[1] - d0[1] * e[0], float(d0 @ e))

                nxt.sort(key=turn)
            prev, cur = cur, nxt[0]
            used.add((min(prev, cur), max(prev, cur)))
        out.append(lp)
        used |= {(min(a, b), max(a, b)) for a, b in _edges(lp)}
    out.sort(key=lambda lp: -abs(polygon_area(P[lp])))
    return out


def write_geo(path, geometry: MachineGeometry, theta: float = 0.0, n_gap: int = 960, h_max: float | None = None,
              grading: float = 0.2, h: float = 1e-3) -> dict:
    """gmsh geometry of the full machine at rotor angle ``theta``. Returns the physical-region numbers, the polygon
    areas of the slots and the magnetisation angles of the magnets (global frame)."""
    from .interop import cross_section_polygons

    rt, st = geometry.rotor, geometry.stator
    r_gap = 0.5 * (rt.r_out + st.r_in)
    h_gap = TWO_PI * r_gap / n_gap
    h_max = h_max or 6 * h_gap

    def lcf(p):
        return min(h_max, h_gap + grading * abs(math.hypot(*p) - r_gap))

    c, s = math.cos(theta), math.sin(theta)
    Rot = np.array([[c, -s], [s, c]])
    g = _PSLG()
    regions: dict[str, list[list[int]]] = {}
    rotor_holes, stator_holes = [], []
    magnets, slots = {}, {}
    for name, xy, mag in cross_section_polygons(geometry, h):
        is_rotor = name.startswith(("magnet", "pocket"))
        if is_rotor:
            xy = xy @ Rot.T
        lp = g.loop(xy, lcf)
        if is_rotor:
            rotor_holes.append(lp)
        else:
            stator_holes.append(lp)
        if name.startswith("magnet"):
            k = len(magnets)
            magnets[k] = (lp, mag + theta)
        elif name.startswith("slot"):
            slots[int(name[4:])] = (lp, abs(polygon_area(xy)))
        else:
            regions.setdefault("air", []).append(lp)
    n_r = max(8, int(round(TWO_PI * rt.r_out / h_gap)))
    shaft = g.loop(_circle(rt.r_in, max(16, int(round(TWO_PI * rt.r_in / min(h_max, lcf((rt.r_in, 0))))))) @ Rot.T, lcf)
    surface = g.loop(_circle(rt.r_out, n_r) @ Rot.T, lcf)
    # the bore: uniform points plus every polygon vertex lying on the bore circle (the slot openings)
    bore_extra = [math.atan2(y, x) for name, xy, _ in cross_section_polygons(geometry, h)
                  if not name.startswith(("magnet", "pocket")) for x, y in xy
                  if abs(math.hypot(x, y) - st.r_in) < 1e-9]
    bore = g.loop(_circle(st.r_in, n_r, bore_extra), lcf)
    outer = g.loop(_circle(st.r_out, max(48, int(round(TWO_PI * st.r_out / h_max)))), lcf)
    surf: dict[str, list[list[int]]] = {
        "rotor_iron": _boundary_loops([surface, shaft] + rotor_holes, g.pts),
        "gap": _boundary_loops([bore, surface], g.pts),
        "stator_iron": _boundary_loops([outer, bore] + [lp for lp in stator_holes], g.pts),
    }
    # stator iron: the bore-loop edges shared with openings cancel; the rest of the bore stays as the inner loop
    lines: dict = {}
    out = ["// written by pinneapple_simulation.electromagnetics.machines.getdp", "Geometry.Tolerance = 1e-12;"]
    for i, ((x, y), lc) in enumerate(zip(g.pts, g.lc, strict=True)):
        out.append(f"Point({i + 1}) = {{{x:.15g}, {y:.15g}, 0, {lc:.6g}}};")

    def line(a, b):
        key = (min(a, b), max(a, b))
        if key not in lines:
            lines[key] = len(lines) + 1
            out.append(f"Line({lines[key]}) = {{{key[0] + 1}, {key[1] + 1}}};")
        return lines[key] if a < b else -lines[key]

    loop_id = [0]

    def curve_loop(lp):
        ids = [line(lp[k], lp[(k + 1) % len(lp)]) for k in range(len(lp))]
        loop_id[0] += 1
        out.append(f"Curve Loop({loop_id[0]}) = {{{', '.join(map(str, ids))}}};")
        return loop_id[0]

    surf_id = [0]

    def plane(loops):
        ids = [curve_loop(lp) for lp in loops]
        surf_id[0] += 1
        out.append(f"Plane Surface({surf_id[0]}) = {{{', '.join(map(str, ids))}}};")
        return surf_id[0]

    phys = {"rotor_iron": 1, "stator_iron": 2, "air": 3, "gap": 4}
    tags = {k: [plane(v)] for k, v in surf.items()}
    tags["air"] = [plane([lp]) for lp in regions.get("air", [])]
    mag_tags, slot_tags = {}, {}
    for k, (lp, _) in magnets.items():
        mag_tags[k] = plane([lp])
    for k, (lp, _) in slots.items():
        slot_tags[k] = plane([lp])
    for name, ids in tags.items():
        out.append(f"Physical Surface({phys[name]}) = {{{', '.join(map(str, ids))}}};")
    for k, t in mag_tags.items():
        out.append(f"Physical Surface({100 + k}) = {{{t}}};")
    for k, t in slot_tags.items():
        out.append(f"Physical Surface({1000 + k}) = {{{t}}};")
    bnd = [abs(line(outer[k], outer[(k + 1) % len(outer)])) for k in range(len(outer))]
    bnd += [abs(line(shaft[k], shaft[(k + 1) % len(shaft)])) for k in range(len(shaft))]
    out.append(f"Physical Curve(10) = {{{', '.join(map(str, bnd))}}};")
    out.append("Mesh.Algorithm = 6;")
    out.append("Mesh.CharacteristicLengthFromPoints = 1; Mesh.CharacteristicLengthExtendFromBoundary = 1;")
    Path(path).write_text("\n".join(out) + "\n")
    return {"magnets": {k: m for k, (_, m) in magnets.items()}, "slot_areas": {k: a for k, (_, a) in slots.items()},
            "r_gap": (rt.r_out, st.r_in), "n_points": len(g.pts)}


NU_TABLE_POINTS = 300  # samples of nu(|B|^2) in the GetDP interpolation table, uniform in |B| up to 1.15 B_max


def write_pro(path, geometry: MachineGeometry, info: dict, steel, Br: float, mu_r: float, layout: list,
              turns_per_slot: float, currents, parallel_paths: int = 1) -> None:
    """GetDP problem for the regions of :func:`write_geo`."""
    curve = steel.curve(geometry.stacking_factor)
    b = np.linspace(0.0, float(curve.B[-1]) * 1.15, NU_TABLE_POINTS)[1:]
    nu = curve.nu(b * b)
    b2 = np.concatenate([[0.0], b * b])
    nu = np.concatenate([[float(curve.nu_init)], nu])
    table = ", ".join(f"{x:.10g}, {y:.10g}" for x, y in zip(b2, nu, strict=True))
    L = geometry.length
    r1, r2 = info["r_gap"]
    mags = info["magnets"]
    slots = info["slot_areas"]
    o = []
    w = o.append
    w("Group {")
    w("  RotorIron = Region[1]; StatorIron = Region[2]; Air = Region[3]; Gap = Region[4]; Outer = Region[10];")
    w(f"  Magnets = Region[{{{', '.join(str(100 + k) for k in mags)}}}];")
    w(f"  Coils = Region[{{{', '.join(str(1000 + k) for k in slots)}}}];")
    w("  Iron = Region[{RotorIron, StatorIron}];")
    w("  DomainL = Region[{Air, Gap, Magnets, Coils}]; DomainNL = Region[{Iron}];")
    w("  Domain = Region[{DomainL, DomainNL}];")
    w("}")
    w("Function {")
    w(f"  mu0 = {MU0:.17g};")
    w("  nu[Region[{Air, Gap, Coils}]] = 1/mu0;")
    w(f"  nu[Magnets] = 1/(mu0*{mu_r:.10g});")
    w(f"  nu_fe[] = InterpolationAkima[$1]{{List[{{{table}}}]}};")
    w(f"  dnu_fe[] = dInterpolationAkima[$1]{{List[{{{table}}}]}};")
    w("  nu[Iron] = nu_fe[SquNorm[$1]];")
    w("  dhdb_NL[Iron] = 2*dnu_fe[SquNorm[$1]]*SquDyadicProduct[$1];")
    for k, ang in mags.items():
        w(f"  br[Region[{100 + k}]] = Vector[{Br * math.cos(ang):.12g}, {Br * math.sin(ang):.12g}, 0];")
    for k, area in slots.items():
        ph, sg = layout[k]
        J = sg * turns_per_slot * currents[ph] / (parallel_paths * area)
        w(f"  js[Region[{1000 + k}]] = Vector[0, 0, {J:.12g}];")
        for q in range(3):
            wq = sg * turns_per_slot / (parallel_paths * area) if q == ph else 0.0
            w(f"  wind{q}[Region[{1000 + k}]] = {wq:.12g};")
    w("}")
    w("Constraint { { Name a; Case { { Region Outer; Value 0; } } } }")
    w("Jacobian { { Name Vol; Case { { Region All; Jacobian Vol; } } } }")
    w("Integration { { Name I1; Case { { Type Gauss; Case { { GeoElement Triangle; NumberOfPoints 3; } } } } } }")
    w("FunctionSpace { { Name Hcurl_a; Type Form1P;")
    w("  BasisFunction { { Name se; NameOfCoef ae; Function BF_PerpendicularEdge; Support Domain; Entity NodesOf[All]; } }")
    w("  Constraint { { NameOfCoef ae; EntityType NodesOf; NameOfConstraint a; } } } }")
    w("Formulation { { Name MagSta; Type FemEquation;")
    w("  Quantity { { Name a; Type Local; NameOfSpace Hcurl_a; } }")
    w("  Equation {")
    w("    Galerkin { [ nu[{d a}] * Dof{d a}, {d a} ]; In Domain; Jacobian Vol; Integration I1; }")
    w("    Galerkin { JacNL [ dhdb_NL[{d a}] * Dof{d a}, {d a} ]; In DomainNL; Jacobian Vol; Integration I1; }")
    w("    Galerkin { [ -nu[] * br[], {d a} ]; In Magnets; Jacobian Vol; Integration I1; }")
    w("    Galerkin { [ -js[], {a} ]; In Coils; Jacobian Vol; Integration I1; }")
    w("  } } }")
    w("Resolution { { Name MagSta; System { { Name A; NameOfFormulation MagSta; } }")
    w("  Operation { InitSolution[A]; IterativeLoop[60, 1e-9, 1]{ GenerateJac[A]; SolveJac[A]; } SaveSolution[A]; } } }")
    w("PostProcessing { { Name Post; NameOfFormulation MagSta; Quantity {")
    w(f"  {{ Name T; Value {{ Integral {{ [ {L:.12g}/(mu0*{r2 - r1:.12g}) * (CompX[{{d a}}]*X[] + CompY[{{d a}}]*Y[]) *"
      f" (CompY[{{d a}}]*X[] - CompX[{{d a}}]*Y[]) / Sqrt[X[]^2 + Y[]^2] ]; In Gap; Jacobian Vol; Integration I1; }} }} }}")
    for q in range(3):
        w(f"  {{ Name psi{q}; Value {{ Integral {{ [ {L:.12g} * wind{q}[] * CompZ[{{a}}] ]; In Coils; Jacobian Vol;"
          f" Integration I1; }} }} }}")
    w("} } }")
    w("PostOperation { { Name Out; NameOfPostProcessing Post; Operation {")
    w('  Print[ T[Gap], OnGlobal, Format Table, File "T.txt" ];')
    for q in range(3):
        w(f'  Print[ psi{q}[Coils], OnGlobal, Format Table, File "psi{q}.txt" ];')
    w("} } }")
    Path(path).write_text("\n".join(o) + "\n")


def _last_number(path: Path) -> float:
    return float(path.read_text().split()[-1])


def solve(geometry: MachineGeometry, theta: float, currents, steel, Br: float, mu_r: float, layout: list,
          turns_per_slot: float, n_gap: int = 960, workdir=None, keep: bool = False, timeout: float = 600.0) -> dict:
    """Mesh with gmsh and solve with GetDP; returns torque (N m), phase flux linkages (Wb), element count, times."""
    if not available():
        raise GmshGetDPError("gmsh and getdp executables are required")
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="getdp_"))
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info = write_geo(tmp / "m.geo", geometry, theta, n_gap)
    r = subprocess.run(["gmsh", "-2", "m.geo", "-format", "msh22", "-o", "m.msh", "-v", "2"], cwd=tmp,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode or not (tmp / "m.msh").exists():
        raise GmshGetDPError("gmsh failed: " + r.stderr[-2000:] + r.stdout[-2000:])
    t1 = time.perf_counter()
    write_pro(tmp / "m.pro", geometry, info, steel, Br, mu_r, layout, turns_per_slot, currents)
    r = subprocess.run(["getdp", "m.pro", "-msh", "m.msh", "-solve", "MagSta", "-pos", "Out", "-v", "3"], cwd=tmp,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode or not (tmp / "T.txt").exists():
        raise GmshGetDPError("getdp failed: " + r.stderr[-3000:] + r.stdout[-3000:])
    t2 = time.perf_counter()
    lines = (tmp / "m.msh").read_text().split("$Elements\n")[1].split("$EndElements")[0].splitlines()[1:]
    n_el = sum(1 for ln in lines if ln.split()[1] == "2")
    out = {"torque": _last_number(tmp / "T.txt"), "psi": np.array([_last_number(tmp / f"psi{q}.txt") for q in range(3)]),
           "elements": n_el, "mesh_s": t1 - t0, "solve_s": t2 - t1, "workdir": str(tmp)}
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
