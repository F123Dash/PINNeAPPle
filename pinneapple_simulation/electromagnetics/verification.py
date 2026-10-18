"""Verification suite of the electromagnetics package: each case runs a solver or a reference implementation
against an independent result (closed form, a published value, another library, a limit) and reports the
measured error and, where a mesh or step is refined, the measured order of convergence.

``run()`` returns :class:`CaseResult` objects; ``to_markdown()`` renders the table of
docs/guides/electromagnetics.md, which is produced by ``examples/electrodynamics/07_em_verification.py``, so every
number in that table comes from a run. ``tests/test_em_verification.py`` asserts each case's tolerance.

Cases are registered with :func:`case`; later modules (FDTD, edge elements, particles, ...) add theirs.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz

__all__ = ["CaseResult", "CASES", "case", "run", "to_markdown", "observed_orders"]


@dataclass
class CaseResult:
    name: str
    what: str                     # quantity compared
    reference: str                # against what
    error: float                  # final (finest) relative error, or the measured deviation
    tolerance: float
    orders: list[float] = field(default_factory=list)   # observed orders between successive refinements
    expected_order: float | None = None
    detail: str = ""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        ok = bool(np.isfinite(self.error) and self.error <= self.tolerance)
        if self.expected_order is not None and self.orders:
            ok = ok and abs(self.orders[-1] - self.expected_order) <= 0.15 * self.expected_order
        return ok

    def row(self) -> str:
        order = ", ".join(f"{o:.2f}" for o in self.orders) if self.orders else "-"
        exp = f" (expected {self.expected_order:g})" if self.expected_order is not None else ""
        cell = lambda t: t.replace("|", "\\|")  # noqa: E731  (|B| inside a Markdown table)
        return (f"| {cell(self.name)} | {cell(self.what)} | {cell(self.reference)} | {self.error:.2e} | "
                f"{order}{exp} | {'pass' if self.passed else 'FAIL'} |")


CASES: dict[str, Callable[[bool], CaseResult]] = {}


def case(name: str):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def observed_orders(errors: Sequence[float], ratio: float = 2.0) -> list[float]:
    e = np.asarray(errors, float)
    return [float(v) for v in np.log(e[:-1] / e[1:]) / math.log(ratio)]


def run(names: Sequence[str] | None = None, quick: bool = False) -> list[CaseResult]:
    out = []
    for n in names or list(CASES):
        t0 = time.perf_counter()
        r = CASES[n](quick)
        r.seconds = time.perf_counter() - t0
        out.append(r)
    return out


def to_markdown(results: Sequence[CaseResult]) -> str:
    head = ("| Case | Quantity | Reference | Error (finest) | Observed order | Status |\n"
            "|---|---|---|---|---|---|")
    return "\n".join([head] + [r.row() for r in results])


# ============================================================================================ statics
@case("Coaxial capacitor (P1 FEM)")
def _coax(quick: bool) -> CaseResult:
    from .electrostatics import capacitance, solve_electrostatic
    from .mesh import Ring, polar_mesh
    from .statics import coax_capacitance

    a, b, eps_r = 1e-3, 3.5e-3, 2.1
    errs = []
    for n in ((16, 32, 64) if quick else (16, 32, 64, 128)):
        rings = [Ring(a + (b - a) * k / (n // 4), n) for k in range(n // 4 + 1)]
        m = polar_mesh(rings, lambda ly, r, p: "dielectric")
        s = solve_electrostatic(m, eps_r, {"inner": (m.rings[0], 1.0), "outer": (m.rings[-1], 0.0)})
        errs.append(abs(capacitance(s, 1.0) / coax_capacitance(a, b, eps_r) - 1))
    return CaseResult("Coaxial capacitor (P1 FEM)", "capacitance per metre", "2 pi eps / ln(b/a)", errs[-1],
                      2e-3, observed_orders(errs), 2.0, f"errors {', '.join(f'{e:.2e}' for e in errs)}")


@case("Two-dielectric parallel-plate capacitor (P1 FEM)")
def _plates(quick: bool) -> CaseResult:
    from pinneapple_physics.closed_form.electromagnetics import two_dielectric_capacitor

    from .electrostatics import solve_electrostatic
    from .mesh import rect_mesh

    d = 1e-3
    m = rect_mesh(np.linspace(0, 0.01, 7), np.linspace(0, d, 9), lambda x, y: "a" if y < d / 2 else "b")
    y = m.points[:, 1]
    s = solve_electrostatic(m, np.where(m.mask("a"), 1.0, 4.0), {"g": (np.flatnonzero(y == 0), 0.0),
                                                                 "p": (np.flatnonzero(y == d), 1.0)})
    err = float(np.max(np.abs(s.phi - two_dielectric_capacitor(y, 1.0, d, d / 2, 1.0, 4.0))))
    return CaseResult("Two-dielectric parallel-plate capacitor (P1 FEM)", "potential (max abs, V)",
                      "D uniform, E_i = D / eps_i (exact in P1)", err, 1e-12)


@case("Circular loop by Biot-Savart (polygon)")
def _loop(quick: bool) -> CaseResult:
    from .statics import biot_savart, circle_segments, loop_field

    pts = np.array([[0.03, 0.01, 0.04], [0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.12, -0.05, -0.02]])
    exact = loop_field(pts, 0.1, 2.0)
    errs = []
    for n in (45, 90, 180, 360):
        s, e = circle_segments(0.1, n)
        errs.append(float(np.max(np.abs(biot_savart(pts, s, e, 2.0) - exact)) / np.max(np.abs(exact))))
    return CaseResult("Circular loop by Biot-Savart (polygon)", "B off the axis",
                      "elliptic-integral loop field (Jackson 5.37)", errs[-1], 1e-4, observed_orders(errs), 2.0)


@case("Infinite wire and finite solenoid (Biot-Savart)")
def _solenoid(quick: bool) -> CaseResult:
    from .statics import biot_savart, finite_solenoid_on_axis_B, helix_segments, infinite_wire_B

    p = np.array([[0.02, 0.01, 0.0], [-0.05, 0.03, 0.1]])
    Bw = infinite_wire_B(p, 3.0)
    e_wire = float(np.max(np.abs(biot_savart(p, [[0, 0, -1e4]], [[0, 0, 1e4]], 3.0) - Bw)) / np.max(np.abs(Bw)))
    z = np.array([0.0, 0.03, 0.05, 0.08])
    sheet = finite_solenoid_on_axis_B(z, 0.02, 0.1, 200, 1.5)
    s, e = helix_segments(0.02, 0.1, 200, n_per_turn=96)
    B = biot_savart(np.column_stack([0 * z, 0 * z, z]), s, e, 1.5)
    err = float(np.max(np.abs(B[:, 2] / sheet - 1)))
    return CaseResult("Infinite wire and finite solenoid (Biot-Savart)", "B of a 200-turn helix on the axis",
                      "current-sheet solenoid (closed form)", err, 5e-3,
                      detail=f"20 km segment vs mu0 I / (2 pi s): {e_wire:.1e}")


@case("Coaxial mutual inductance")
def _mutual(quick: bool) -> CaseResult:
    from scipy.integrate import quad

    from .constants import MU0
    from .statics import mutual_inductance_coaxial_loops

    r1, r2, d = 0.1, 0.07, 0.05
    M = MU0 / 2 * quad(lambda t: r1 * r2 * math.cos(t) / math.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * math.cos(t)
                                                                   + d * d), 0, 2 * math.pi, epsabs=1e-14)[0]
    err = abs(mutual_inductance_coaxial_loops(r1, r2, d) / M - 1)
    return CaseResult("Coaxial mutual inductance", "M (elliptic integrals)", "Neumann double integral", err, 1e-9)


@case("Axisymmetric coil (FDM, eddy_current_fdm at DC)")
def _coil(quick: bool) -> CaseResult:
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
        MU0,
        annular_current_source,
        axial_flux_density,
        solve_axisymmetric_eddy_current,
    )

    errs = []
    for n in ((121, 241) if quick else (121, 241, 481)):
        r = np.linspace(0, 0.6, n)
        z = np.linspace(-0.6, 0.6, 2 * n - 1)
        J = annular_current_source(r, z, 0.03, 0.0, 0.004, 0.002, 10.0, 0.5)
        dr, dz = r[1] - r[0], z[1] - z[0]
        RR = np.ones((n, len(z)))
        A = solve_axisymmetric_eddy_current(r, z, 0.0, MU0 * RR, 0 * RR, J)
        Bz = axial_flux_density(A, r, z).real
        k = int(np.argmin(np.abs(z - 0.05)))
        ii, jj = np.nonzero(J)
        ref = np.sum(J[ii, jj] * dr * dz * MU0 * r[ii] ** 2 / (2 * (r[ii] ** 2 + (z[jj] - z[k]) ** 2) ** 1.5))
        errs.append(abs(Bz[0, k] / ref - 1))
    return CaseResult("Axisymmetric coil (FDM, eddy_current_fdm at DC)", "on-axis B_z",
                      "sum of current-loop fields (closed form)", errs[-1], 2e-3,
                      detail=f"errors {', '.join(f'{e:.2e}' for e in errs)} (finite box: A = 0 at 20 coil radii)")


# ============================================================================================ waves
@case("Mie sphere vs Bohren & Huffman / miepython")
def _mie(quick: bool) -> CaseResult:
    from .mie import mie

    r = mie(1.55, 2 * math.pi * 0.525 / 0.6328)
    pub = (3.10543, 3.10543, 2.92534)    # BHMIE sample output, Bohren & Huffman (1983) appendix A
    err = max(abs(r.qsca - pub[0]) / pub[0], abs(r.qext - pub[1]) / pub[1], abs(r.qback - pub[2]) / pub[2])
    detail = "published 5-digit values"
    try:
        import miepython

        e2 = 0.0
        for m, x in [(1.55, 5.213), (1.5 - 0.1j, 3.0), (1.33, 50.0), (4 - 3j, 1.2)]:
            qe, qs, qb, _ = miepython.efficiencies_mx(m, x)
            rr = mie(m, x)
            e2 = max(e2, abs(rr.qext / qe - 1), abs(rr.qsca / qs - 1), abs(rr.qback / qb - 1))
        detail += f"; miepython {miepython.__version__}, 4 spheres incl. absorbing: max rel. diff {e2:.1e}"
    except ImportError:
        detail += "; miepython not installed"
    return CaseResult("Mie sphere vs Bohren & Huffman / miepython", "Q_sca, Q_ext, Q_back",
                      "BHMIE sample (B&H app. A)", err, 2e-5, detail=detail)


@case("Mie small-sphere (Rayleigh) limit")
def _rayleigh(quick: bool) -> CaseResult:
    from .mie import mie, mie_pec, rayleigh_qsca

    xs = (0.1, 0.05, 0.025)
    errs = [abs(mie(1.5, x).qsca / rayleigh_qsca(1.5, x) - 1) for x in xs]
    pec = [abs(mie_pec(x).qsca / (10 / 3 * x ** 4) - 1) for x in xs]
    return CaseResult("Mie small-sphere (Rayleigh) limit", "Q_sca / Rayleigh - 1 vs x",
                      "8/3 x^4 |(m^2-1)/(m^2+2)|^2", errs[-1], 1e-4, observed_orders(errs), 2.0,
                      f"PEC sphere vs 10/3 x^4: orders {', '.join(f'{o:.2f}' for o in observed_orders(pec))}")


@case("Fresnel coefficients and Brewster angle")
def _fresnel(quick: bool) -> CaseResult:
    from .waves import brewster_angle, fresnel

    worst = 0.0
    for n1, n2 in ((1.0, 1.5), (1.0, 3.4), (1.5, 1.0)):
        th = np.linspace(0, 1.5, 61)
        rs, rp, ts, tp = fresnel(n1, n2, th)
        s = n1 * np.sin(th) / n2
        ok = s < 1
        f = (n2 * np.sqrt(1 - s[ok] ** 2 + 0j)).real / (n1 * np.cos(th[ok]))
        worst = max(worst, float(np.max(np.abs(np.abs(rs[ok]) ** 2 + f * np.abs(ts[ok]) ** 2 - 1))),
                    float(np.max(np.abs(np.abs(rp[ok]) ** 2 + f * np.abs(tp[ok]) ** 2 - 1))))
    rb = abs(fresnel(1.0, 1.5, brewster_angle(1.0, 1.5))[1])
    return CaseResult("Fresnel coefficients and Brewster angle", "R + T - 1 (s and p, 3 interfaces)",
                      "energy conservation; r_p(atan n2/n1) = 0", max(worst, rb), 1e-12,
                      detail=f"|r_p| at Brewster {rb:.1e}")


@case("Bragg mirror (characteristic matrices)")
def _bragg(quick: bool) -> CaseResult:
    from .waves import multilayer

    lam, nH, nL, N, ns = 1e-6, 2.3, 1.38, 6, 1.52
    stack = [nH, nL] * N
    R = multilayer(stack, [lam / (4 * n) for n in stack], lam, n_out=ns).R
    Y = (nH / nL) ** (2 * N) * ns
    err = abs(R / ((1 - Y) / (1 + Y)) ** 2 - 1)
    return CaseResult("Bragg mirror (characteristic matrices)", "R at the design wavelength",
                      "((1 - Y)/(1 + Y))^2, Y = (nH/nL)^2N ns", err, 1e-12)


@case("Hertzian dipole radiated power")
def _hertz(quick: bool) -> CaseResult:
    from .constants import C0
    from .waves import hertzian_dipole_fields, hertzian_dipole_power

    omega = 2 * math.pi * 1e9
    k = omega / C0
    err = 0.0
    for r in (0.05 / k, 0.5 / k, 20 / k):
        th = np.linspace(0, math.pi, 2001)
        _, Eth, Hph = hertzian_dipole_fields(r, th, 1e-3, omega)
        P = _trapz(0.5 * np.real(Eth * np.conj(Hph)) * 2 * math.pi * r * r * np.sin(th), th)
        err = max(err, abs(P / hertzian_dipole_power(1e-3, omega) - 1))
    return CaseResult("Hertzian dipole radiated power", "flux of the complete fields at kr = 0.05, 0.5, 20",
                      "eta k^2 |I l|^2 / (12 pi)", err, 1e-6)


@case("Half-wave dipole radiation resistance")
def _halfwave(quick: bool) -> CaseResult:
    from .constants import ETA0
    from .waves import dipole_pattern, half_wave_dipole_radiation_resistance

    th = np.linspace(1e-9, math.pi - 1e-9, 200001)
    R = 2 * ETA0 / (4 * math.pi) * _trapz(dipole_pattern(th, 0.5) ** 2 * np.sin(th), th)
    Rc = half_wave_dipole_radiation_resistance()
    return CaseResult("Half-wave dipole radiation resistance", "R_rad from the pattern",
                      "eta Cin(2 pi) / (4 pi) = 73.08 ohm", abs(R / Rc - 1), 1e-6, detail=f"{Rc:.3f} ohm")


@case("Dispersive ADE recursions (Drude, Debye, Lorentz)")
def _ade(quick: bool) -> CaseResult:
    from .constants import EPS0
    from .media import Debye, Drude, Lorentz

    d = Drude(1e15, 1e14)
    errs = []
    for dt in (1e-16, 5e-17, 2.5e-17):
        ka, kb = d.ade(dt)
        z = np.exp(1j * 3e14 * dt)
        jw = 2 / dt * (1 - 1 / z) / (1 + 1 / z)
        errs.append(abs(kb * (1 + 1 / z) / (1 - ka / z) / (jw * EPS0) / d.chi(3e14) - 1))
    deb = Debye(70.0, 8e-12)
    e_deb = []
    for dt in (1e-12, 5e-13, 2.5e-13):
        ka, kb = deb.ade(dt)
        z = np.exp(1j * 2e10 * dt)
        e_deb.append(abs(kb * (1 + 1 / z) / (1 - ka / z) / EPS0 / deb.chi(2e10) - 1))
    lz = Lorentz(1.5, 4e15, 1e14)
    e_lz = []
    for dt in (2e-17, 1e-17, 5e-18):
        a1, a2, a3 = lz.ade(dt)
        z = np.exp(1j * 1e15 * dt)
        e_lz.append(abs(a3 / (1 - a1 / z - a2 / z ** 2) / z / EPS0 / lz.chi(1e15) - 1))
    det = (f"Debye orders {', '.join(f'{o:.2f}' for o in observed_orders(e_deb))}; Lorentz "
           f"{', '.join(f'{o:.2f}' for o in observed_orders(e_lz))}")
    return CaseResult("Dispersive ADE recursions (Drude, Debye, Lorentz)", "discrete chi(omega) vs dt (Drude)",
                      "continuous chi(omega)", errs[-1], 1e-4, observed_orders(errs), 2.0, det)


@case("Uniaxial crystal indices")
def _uniaxial(quick: bool) -> CaseResult:
    from .media import Anisotropic

    no, ne = 1.5, 1.7
    u = Anisotropic.uniaxial(no, ne)
    err = 0.0
    for th in np.linspace(0, math.pi / 2, 9):
        n = sorted(u.indices([math.sin(th), 0, math.cos(th)]).real)
        ext = 1 / math.sqrt(math.cos(th) ** 2 / no ** 2 + math.sin(th) ** 2 / ne ** 2)
        err = max(err, float(np.max(np.abs(np.array(n) - sorted([no, ext])))))
    return CaseResult("Uniaxial crystal indices", "n_o, n_e(theta) from the wave-normal eigenproblem",
                      "index ellipsoid", err, 1e-12)


# ============================================================================================ magnetics
@case("Round wire, A_z P1 FEM")
def _wire(quick: bool) -> CaseResult:
    from .analytic import wire_field
    from .fem import Geometry, solve_magnetostatic
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    a, R, cur = 0.01, 0.05, 100.0
    errs = []
    for n in (32, 64, 128):
        radii = np.unique(np.concatenate([np.linspace(a / (n // 8), a, n // 8), np.linspace(a, R, n // 4)]))
        mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "cu" if r < a else "air")
        g = Geometry.of(mesh)
        J = np.where(mesh.mask("cu"), 1.0, 0.0)
        J *= cur / np.sum(J * g.area)
        sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J)
        A_ex, _ = wire_field(np.hypot(*mesh.points.T), cur, a, R)
        errs.append(float(np.abs(sol.A - A_ex).max() / np.abs(A_ex).max()))
    return CaseResult("Round wire, A_z P1 FEM", "A_z (max relative)", "Ampere's law (closed form)", errs[-1], 1e-3,
                      observed_orders(errs), 2.0)


@case("Skin effect in a round wire (A_z eddy FEM)")
def _skin(quick: bool) -> CaseResult:
    from .analytic import skin_effect_wire
    from .constants import MU0
    from .eddy import solve_time_harmonic
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    a, R, sigma = 0.005, 0.02, 5.8e7
    delta = a / 3
    omega = 2 / (MU0 * sigma * delta ** 2)
    n = 128
    radii = np.concatenate([np.linspace(a / (n // 4), a, n // 4), np.geomspace(a, R, n // 8)[1:]])
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii], lambda ly, r, p: "cu" if r < a else "air")
    m = mesh.mask("cu")
    sol = solve_time_harmonic(mesh, np.full(mesh.n_elements, NU0), np.full(mesh.n_elements, sigma), omega, [m], [1.0])
    _, Z = skin_effect_wire(0.0, 1.0, a, sigma, omega)
    Zt = Z + 1j * omega * MU0 / (2 * math.pi) * math.log(R / a)
    err = max(abs(sol.u[0].real / Zt.real - 1), abs(sol.u[0].imag / Zt.imag - 1))
    return CaseResult("Skin effect in a round wire (A_z eddy FEM)", "impedance R + j X at a / delta = 3",
                      "Bessel solution (closed form)", err, 3e-3)


@case("Froehlich-Kennelly iron ring (Newton)")
def _fk(quick: bool) -> CaseResult:
    from .fem import Geometry, solve_magnetostatic
    from .hysteresis import FrohlichKennelly
    from .materials import NU0
    from .mesh import Ring, polar_mesh

    fk = FrohlichKennelly.from_saturation(3000.0, 1.9)
    a, r1, r2, R, n = 0.005, 0.01, 0.02, 0.03, 64
    radii = np.unique(np.concatenate([np.linspace(a / 4, a, 4), np.linspace(a, R, n // 2)[1:], [r1, r2]]))
    mesh = polar_mesh([Ring(0.0, 1)] + [Ring(r, n) for r in radii],
                      lambda ly, r, p: "cu" if r < a else ("fe" if r1 < r < r2 else "air"))
    g = Geometry.of(mesh)
    J = np.where(mesh.mask("cu"), 1.0, 0.0)
    J *= 3000.0 / np.sum(J * g.area)
    sol = solve_magnetostatic(mesh, np.full(mesh.n_elements, NU0), J=J, nonlinear=[(mesh.mask("fe"), fk)])
    c = mesh.centroids()[mesh.mask("fe")]
    H = 3000.0 / (2 * math.pi * np.hypot(c[:, 0], c[:, 1]))
    err = float(np.max(np.abs(np.hypot(*sol.B[mesh.mask("fe")].T) / fk.b(H) - 1)))
    return CaseResult("Froehlich-Kennelly iron ring (Newton)", "|B| in saturated iron", "B(H) at H = I / (2 pi r)",
                      err, 0.03, detail=f"Newton: {sol.iterations} iterations to residual {sol.residual:.1e}")


@case("Jiles-Atherton integrator")
def _ja(quick: bool) -> CaseResult:
    from .hysteresis import JilesAtherton

    ja = JilesAtherton(1.6e6, 1100.0, 400.0, 0.2, 1.6e-3)
    H = np.concatenate([np.linspace(0, 5e3, 101), np.linspace(5e3, -5e3, 201)[1:], np.linspace(-5e3, 5e3, 201)[1:]])
    ref = ja.trace(H, substeps=256).M
    errs = [float(np.max(np.abs(ja.trace(H, substeps=s).M - ref)) / ja.Ms) for s in (2, 4, 8, 16, 32)]
    rev = JilesAtherton(1.6e6, 1100.0, 400.0, 1.0, 1.6e-3)
    lr = rev.trace(H)
    dev = float(np.max(np.abs(lr.M - rev.anhysteretic(lr.H + rev.alpha * lr.M))) / rev.Ms)
    return CaseResult("Jiles-Atherton integrator", "M along a major loop vs RK4 sub-steps",
                      "256 sub-steps; c = 1 stays anhysteretic", errs[-1], 1e-8, observed_orders(errs),
                      detail=f"c = 1 deviation from the anhysteretic curve {dev:.1e} Ms (orders irregular: the "
                             "pinning switch makes dM/dH non-smooth)")


# ============================================================================================ checks
@case("Maxwell residual checks on an exact cavity mode")
def _checks(quick: bool) -> CaseResult:
    from pinneapple_physics import maxwell_checks as mc

    C = 299_792_458.0
    kx, ky = math.pi, 2 * math.pi
    w = C * math.hypot(kx, ky)
    vals, bad = [], None
    for n in (21, 41, 81):
        x = y = np.linspace(0, 1, n)
        t = np.linspace(0, 0.7 * 2 * math.pi / w, n)
        X, Y, T = np.meshgrid(x, y, t, indexing="ij")

        def fields(wf):
            return {"Ez": np.sin(kx * X) * np.sin(ky * Y) * np.cos(wf * T),
                    "Bx": -ky * np.sin(kx * X) * np.cos(ky * Y) * np.sin(wf * T) / wf,
                    "By": kx * np.cos(kx * X) * np.sin(ky * Y) * np.sin(wf * T) / wf}
        ch = {c.key: c for c in mc.maxwell_checks({"x": x, "y": y, "t": t}, fields(w))}
        vals.append(ch["ampere_maxwell"].value)
        if n == 41:
            bad = {c.key: c for c in mc.maxwell_checks({"x": x, "y": y, "t": t}, fields(1.2 * w))}
    return CaseResult("Maxwell residual checks on an exact cavity mode", "Ampere-Maxwell residual vs grid",
                      "exact TM12 mode (residual -> 0)", vals[-1], 1e-3, observed_orders(vals), 2.0,
                      f"a 20 % wrong frequency gives {bad['ampere_maxwell'].value:.3f} (rejected) and Poynting "
                      f"imbalance {bad['poynting_balance'].value:.3f}")
