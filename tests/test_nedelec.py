"""Edge (Nedelec / Whitney) elements: the structural properties that make them spurious-free, mesh conformity,
the eigen and driven solvers, the scikit-fem backend, and the induction-heating additions to eddy_current_fdm.
Accuracy against closed forms (guides, cavities, slab-loaded cavity, manufactured solutions, PML line, billet) is in
the verification suite (tests/test_em_verification.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg as sl

from pinneapple_simulation.electromagnetics import nedelec as ned
from pinneapple_simulation.electromagnetics.mesh import Ring, polar_mesh, rect_mesh


def small_2d():
    m = rect_mesh(np.linspace(0, 1, 6), np.linspace(0, 0.6, 4))
    return m, ned.EdgeSpace.of(m.points, m.tris)


def test_whitney_degrees_of_freedom_are_dual_to_the_basis():
    """int_e N_a . t_e dl = delta_ae on every edge: the line-integral DOFs of each basis function are a unit vector
    (exact solution of the interpolation problem), in 2-D and 3-D."""
    for space in (small_2d()[1], ned.EdgeSpace.of(*(lambda t: (t.points, t.tets))(
            ned.tet_box_mesh(np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))))):
        for a in (0, space.n_edges // 3, space.n_edges - 1):
            # the field of basis function a, evaluated at points on each edge through the containing element
            x = np.zeros(space.n_edges)
            x[a] = 1.0

            def field(X, x=x, space=space):
                return np.real(space.evaluate(x, X))

            dofs = ned.edge_dofs(space, field)
            np.testing.assert_allclose(dofs, x, atol=1e-9)


def test_gradients_are_in_the_kernel_of_the_curl():
    """S G = 0 exactly (the discrete de Rham property): curl of the gradient of any nodal function vanishes."""
    _, space = small_2d()
    S, T = ned.assemble(space)
    G = ned.gradient_matrix(space)
    assert abs(S @ G).max() < 1e-12 * abs(S).max()
    tm = ned.tet_box_mesh(*(np.linspace(0, 1, 4),) * 3)
    s3 = ned.EdgeSpace.of(tm.points, tm.tets)
    S3, _ = ned.assemble(s3)
    assert abs(S3 @ ned.gradient_matrix(s3)).max() < 1e-12 * abs(S3).max()


def test_no_spurious_modes_null_space_is_exactly_the_interior_gradients():
    """Dense spectrum of a PEC rectangle: exactly (number of interior nodes) zero eigenvalues, then the analytic
    TE cut-off pi^2 (first non-zero), nothing in between."""
    m, space = small_2d()
    S, T = ned.assemble(space)
    fr = np.setdiff1d(np.arange(space.n_edges), ned.boundary_edges(space))
    w = sl.eigh(S[fr][:, fr].toarray(), T[fr][:, fr].toarray(), eigvals_only=True)
    from pinneapple_simulation.electromagnetics.mesh import boundary_nodes

    n_interior = len(m.points) - len(boundary_nodes(m.tris))
    zero = np.abs(w) < 1e-8 * w.max()
    assert zero.sum() == n_interior
    assert w[~zero].min() == pytest.approx(math.pi ** 2, rel=0.05)
    # the constrained sparse solver returns the same non-zero spectrum
    np.testing.assert_allclose(ned.eigenmodes(space, 4), np.sort(w[~zero])[:4], rtol=1e-9)


def test_tet_meshes_are_conforming():
    """Every interior triangle face is shared by exactly two tetrahedra and the volumes add up, for the box mesher and
    for the extruded polar mesh (minimum-vertex-index prism split)."""
    def check(tm, volume):
        f = np.sort(np.concatenate([tm.tets[:, [0, 1, 2]], tm.tets[:, [0, 1, 3]], tm.tets[:, [0, 2, 3]],
                                    tm.tets[:, [1, 2, 3]]]), axis=1)
        _, counts = np.unique(f, axis=0, return_counts=True)
        assert set(counts) <= {1, 2}
        s = ned.EdgeSpace.of(tm.points, tm.tets)
        assert s.vol.sum() == pytest.approx(volume, rel=1e-12)
        return counts

    check(ned.tet_box_mesh(np.linspace(0, 1, 4), np.linspace(0, 2, 3), np.linspace(0, 0.5, 5)), 1.0)
    rings = [Ring(0.0, 1), Ring(0.5, 8), Ring(1.0, 16)]
    m = polar_mesh(rings, lambda ly, r, p: "air")
    area = ned.EdgeSpace.of(m.points, m.tris).vol.sum()
    counts = check(ned.extrude_to_tets(m.points, m.tris, np.linspace(0, 1, 4)), area)
    assert (counts == 2).sum() > 0


def test_waveguide_modes_rectangle_labels_and_values():
    """TE10 is the dominant mode of a 2:1 rectangle; the first TM mode is TM11 at the TE11 cut-off (closed form)."""
    m = rect_mesh(np.linspace(0, 2, 33), np.linspace(0, 1, 17))
    modes = ned.waveguide_modes(m.points, m.tris, 6)
    assert modes[0][1] == "TE" and modes[0][0] == pytest.approx(math.pi / 2, rel=2e-3)
    tm = [k for k, kind in modes if kind == "TM"]
    assert tm and tm[0] == pytest.approx(math.pi * math.hypot(0.5, 1), rel=5e-3)


def test_driven_solution_matches_manufactured_field_and_pec_values():
    """solve_driven with inhomogeneous PEC values reproduces a manufactured solution exactly when it lies in the
    space: E = (-y, x) = e_z x r has a constant curl, so curl curl E = 0 and the source is -k^2 E."""
    m = rect_mesh(np.linspace(0, 1, 17), np.linspace(0, 1, 17))
    space = ned.EdgeSpace.of(m.points, m.tris)
    k0 = 2.0
    from pinneapple_simulation.electromagnetics.constants import C0

    E = lambda X: np.column_stack([-X[:, 1], X[:, 0]])     # noqa: E731
    bnd = ned.boundary_edges(space)
    x = ned.solve_driven(space, k0 * C0, ned.load_from_function(space, lambda X: -k0 ** 2 * E(X)),
                         pec=bnd, pec_values=ned.edge_dofs(space, E, bnd))
    # a + b x r is in the lowest-order Nedelec space: reproduced to round-off
    eE, ecurl = ned.errors(space, x, E, lambda X: np.full(len(X), 2.0))
    assert eE < 1e-10 and ecurl < 1e-10


def test_scikit_fem_backend_gives_the_same_spectrum():
    """In-house assembly against the reference solution of scikit-fem's ElementTriN1 / ElementTetN0 on the same mesh."""
    pytest.importorskip("skfem")
    from pinneapple_simulation.electromagnetics import backends

    m = rect_mesh(np.linspace(0, 1, 9), np.linspace(0, 0.5, 5))
    a = ned.eigenmodes(ned.EdgeSpace.of(m.points, m.tris), 5)
    b = backends.skfem_eigenmodes(m.points, m.tris, 5)
    np.testing.assert_allclose(a, b, rtol=1e-11)
    tm = ned.tet_box_mesh(np.linspace(0, 1, 5), np.linspace(0, 0.5, 3), np.linspace(0, 0.75, 4))
    eps = np.where(tm.centroids()[:, 0] < 0.4, 4.0, 1.0)
    np.testing.assert_allclose(ned.eigenmodes(ned.EdgeSpace.of(tm.points, tm.tets), 4, eps_r=eps),
                               backends.skfem_eigenmodes(tm.points, tm.tets, 4, eps_r=eps), rtol=1e-11)


def test_pml_tensors_are_identity_inside_and_absorbing_outside():
    c = np.array([[0.0, 0.0], [1.2, 0.0], [0.0, -1.3]])
    eps, mu = ned.pml_tensors(c, [(-1, 1), (-1, 1)], 0.5, 2 * math.pi * 1e9)
    np.testing.assert_allclose(eps[0], np.eye(3))
    assert eps[1][0, 0].imag > 0 and eps[1][1, 1].imag < 0          # 1/s_x and s_x with s_x = 1 - j sigma/(w eps0)
    np.testing.assert_allclose(eps, mu)


def test_eddy_fdm_boundary_values_and_power_closed_form():
    """eddy_current_fdm with boundary values reproduces the analytic solution of a billet in a uniform field, and
    the closed-form power equals the integral of |J|^2 / (2 sigma)."""
    from scipy.integrate import quad
    from scipy.special import jv

    from pinneapple_simulation.electromagnetics.analytic import cylinder_in_axial_field
    from pinneapple_simulation.numerical_solvers.eddy_current_fdm import MU0, solve_axisymmetric_eddy_current

    a, sigma, w, H0 = 0.02, 1e6, 2 * math.pi * 5e3, 1e4
    _, P = cylinder_in_axial_field(0.0, a, sigma, w, H0)
    k = np.sqrt(-1j * w * MU0 * sigma)
    direct = quad(lambda r: abs(w * sigma * MU0 * H0 * jv(1, k * r) / (k * jv(0, k * a))) ** 2 / (2 * sigma)
                  * 2 * math.pi * r, 0, a, limit=200)[0]
    assert P == pytest.approx(direct, rel=1e-8)
    r = np.linspace(0, 0.04, 81)
    z = np.linspace(0, 0.02, 21)
    RR = np.repeat(r[:, None], len(z), axis=1)
    Ar, _ = cylinder_in_axial_field(r, a, sigma, w, H0)
    sig = np.where(RR < a - 1e-12, sigma, 0.0) + np.where(np.abs(RR - a) < 1e-12, 0.5 * sigma, 0.0)
    A = solve_axisymmetric_eddy_current(r, z, w, MU0 * np.ones_like(RR), sig, np.zeros_like(RR),
                                        np.repeat(Ar[:, None], len(z), axis=1))
    assert np.max(np.abs(A - Ar[:, None])) < 1e-3 * np.max(np.abs(Ar))
    # without boundary values the solver still imposes A = 0 (unchanged default)
    A0 = solve_axisymmetric_eddy_current(r, z, w, MU0 * np.ones_like(RR), sig, np.zeros_like(RR))
    assert np.max(np.abs(A0)) == 0.0
