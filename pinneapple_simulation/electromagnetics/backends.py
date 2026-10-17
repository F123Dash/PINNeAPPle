"""Optional finite-element backends for the edge-element solver, detected at import.

``HAVE_SKFEM``: scikit-fem (BSD-3) provides lowest-order Nedelec elements (``ElementTriN1`` on triangles,
``ElementTetN0`` on tetrahedra). :func:`skfem_matrices` assembles the curl-curl and mass matrices with scikit-fem on
the same mesh as :mod:`.nedelec`, and :func:`skfem_eigenmodes` solves the cavity / cut-off problem with the same
divergence-constrained eigensolver; the two assemblies give the same spectrum to round-off (tested), which
cross-checks the in-house element integrals. scikit-fem orients each edge from its lower to its higher vertex index,
as :mod:`.nedelec` does.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - availability depends on the environment
    import skfem  # noqa: F401

    HAVE_SKFEM = True
except ImportError:  # pragma: no cover
    HAVE_SKFEM = False

__all__ = ["HAVE_SKFEM", "skfem_matrices", "skfem_eigenmodes"]


def skfem_matrices(points, cells, eps_r=1.0, mu_r=1.0):
    """``(S, T, edges, boundary_dofs)`` assembled by scikit-fem for scalar per-element ``eps_r`` and ``mu_r``;
    ``edges`` are the (E, 2) node pairs in scikit-fem's DOF order."""
    if not HAVE_SKFEM:
        raise ImportError("scikit-fem is not installed: pip install scikit-fem")
    from skfem import Basis, BilinearForm, ElementTetN0, ElementTriN1, MeshTet, MeshTri
    from skfem.helpers import curl, dot

    pts = np.asarray(points, float)
    cl = np.sort(np.asarray(cells, np.int64), axis=1)
    d = pts.shape[1]
    mesh = (MeshTri if d == 2 else MeshTet)(pts.T.copy(), cl.T.copy())
    basis = Basis(mesh, ElementTriN1() if d == 2 else ElementTetN0())
    M = len(cl)
    eps = np.broadcast_to(np.asarray(eps_r, float), (M,))
    nu = 1.0 / np.broadcast_to(np.asarray(mu_r, float), (M,))
    nq = basis.X.shape[-1]
    eps_q = np.repeat(eps[:, None], nq, axis=1)
    nu_q = np.repeat(nu[:, None], nq, axis=1)

    @BilinearForm
    def cc(E, v, w):
        c1, c2 = curl(E), curl(v)
        return w["nu"] * (c1 * c2 if d == 2 else dot(c1, c2))

    @BilinearForm
    def mm(E, v, w):
        return w["eps"] * dot(E, v)

    S = cc.assemble(basis, nu=nu_q)
    T = mm.assemble(basis, eps=eps_q)
    edges = (mesh.facets if d == 2 else mesh.edges).T
    return S, T, np.sort(edges, axis=1), basis.get_dofs().all()


def skfem_eigenmodes(points, cells, n: int = 6, eps_r=1.0, mu_r=1.0, k_guess: float | None = None):
    """Lowest ``n`` cavity / cut-off eigenvalues ``k^2`` (PEC boundary) from scikit-fem's assembly and the
    divergence-constrained eigensolver of :mod:`.nedelec`."""
    import scipy.sparse as sp

    from .nedelec import constrained_eigs

    S, T, edges, bnd = skfem_matrices(points, cells, eps_r, mu_r)
    E = S.shape[0]
    G = sp.csr_matrix((np.tile([-1.0, 1.0], E), (np.repeat(np.arange(E), 2), edges.ravel())),
                      shape=(E, len(np.asarray(points))))
    free = np.setdiff1d(np.arange(E), bnd)
    if k_guess is None:
        k_guess = 0.9 * math.pi / np.ptp(np.asarray(points, float), axis=0).max()
    vals, _ = constrained_eigs(S, T, G, edges, free, n, k_guess ** 2)
    return vals
