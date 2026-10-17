"""Frequency-domain finite elements for the vector wave equation with lowest-order Nedelec (Whitney edge) elements,
on triangles (2-D, in-plane E) and tetrahedra (3-D).

The unknowns are the tangential components of E along the mesh edges. The basis of edge ``(i, j)`` (global node
indices ``i < j``) is Whitney's ``N_ij = l_i grad(l_j) - l_j grad(l_i)`` (``l`` barycentric coordinates): its
tangential component is continuous across elements and its normal component may jump, which is what E does at a
material interface. Gradients of nodal functions lie in the span, so the discrete curl has the right null space and
the eigenvalue problem has no spurious non-zero modes (J. C. Nedelec, Numer. Math. 35 (1980) 315; J.-M. Jin,
*The Finite Element Method in Electromagnetics*, 3rd ed., Wiley-IEEE 2014, ch. 8-9; P. Monk, *Finite Element Methods
for Maxwell's Equations*, Oxford 2003).

Weak form (``exp(+j omega t)``, ``k0 = omega / c``)::

    int (mu_r^-1 curl E) . curl v  -  k0^2 int (eps_r E) . v  =  -j omega mu0 int J . v

with PEC (tangential E = 0) on the boundary edges that are not left free. ``eps_r`` and ``mu_r`` are per element:
scalars or 3x3 tensors (diagonal or full, complex allowed), which covers anisotropic media, losses and the
stretched-coordinate perfectly matched layer (:func:`pml_tensors`; Jin ch. 9). In 2-D the in-plane field gives the
TE (``H_z``) polarisation and the curl is the scalar ``dE_y/dx - dE_x/dy``, weighted by ``1/mu_zz``.

:func:`assemble` returns the curl-curl matrix ``S`` and the mass matrix ``T``;
:func:`eigenmodes` solves ``S x = k^2 T x`` (cavity resonances, TE cut-offs of a waveguide cross-section) with a
shift that skips the null space of gradients; :func:`solve_driven` solves the driven problem;
:meth:`EdgeSpace.evaluate` gives E (and curl E) at arbitrary points.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .constants import C0, EPS0, MU0

__all__ = ["EdgeSpace", "assemble", "eigenmodes", "constrained_eigs", "edge_averages", "gradient_matrix", "load_from_function", "edge_dofs", "errors", "solve_driven", "pml_tensors", "boundary_edges",
           "dipole_load", "tet_box_mesh", "extrude_to_tets", "TetMesh"]

_TRI_EDGES = ((0, 1), (0, 2), (1, 2))
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


# ------------------------------------------------------------------------------------------------ meshes
@dataclass
class TetMesh:
    points: np.ndarray            # (N, 3)
    tets: np.ndarray              # (M, 4)
    region: np.ndarray | None = None

    @property
    def n_elements(self) -> int:
        return len(self.tets)

    def centroids(self) -> np.ndarray:
        return self.points[self.tets].mean(1)


def tet_box_mesh(x, y, z) -> TetMesh:
    """Structured tetrahedral mesh of the box spanned by node coordinates ``x, y, z``: each hexahedron is split into
    the six tetrahedra around its main diagonal (Kuhn), which is conforming across neighbouring cells."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    nx, ny, nz = len(x), len(y), len(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    c = [idx[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k].ravel() for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    v = dict(zip(((i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)), c, strict=True))
    paths = [((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
             ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    tets = [np.column_stack([v[(0, 0, 0)], v[a], v[b], v[(1, 1, 1)]]) for a, b in paths]
    return TetMesh(pts, np.vstack(tets))


def extrude_to_tets(points2d, tris, z) -> TetMesh:
    """Extrude a triangle mesh along z through the levels ``z`` into prisms and split each prism into three
    tetrahedra by the minimum-vertex-index rule, which makes the split of every shared quadrilateral face agree
    between neighbours (conforming for any triangle mesh)."""
    p2 = np.asarray(points2d, float)
    z = np.asarray(z, float)
    n2 = len(p2)
    pts = np.column_stack([np.tile(p2, (len(z), 1)), np.repeat(z, n2)])
    out = []
    for lvl in range(len(z) - 1):
        bot = tris + lvl * n2
        top = tris + (lvl + 1) * n2
        prism = np.concatenate([bot, top], axis=1)          # (a, b, c, a', b', c')
        for p in prism:
            out.extend(_split_prism(p))
    return TetMesh(pts, np.asarray(out, np.int64))


# vertex permutations that bring vertex k of a prism (V0 V1 V2 bottom, V3 V4 V5 above them) to position 0
_PRISM_ROT = ((0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4),
              (3, 5, 4, 0, 2, 1), (4, 3, 5, 1, 0, 2), (5, 4, 3, 2, 1, 0))


def _split_prism(p):
    """Three tetrahedra of a prism, after J. Dompierre, P. Labbe, M.-G. Vallet and R. Camarero, How to subdivide
    pyramids, prisms and hexahedra into tetrahedra, Proc. 8th International Meshing Roundtable (1999): rotate the
    smallest global vertex to V0, then cut the quadrilateral face V1 V2 V5 V4 through the diagonal that contains the
    smaller of its corner indices. Neighbouring prisms make the same choice on a shared face."""
    p = [int(v) for v in p]
    V = [p[i] for i in _PRISM_ROT[int(np.argmin(p))]]
    if min(V[1], V[5]) < min(V[2], V[4]):
        return [[V[0], V[1], V[2], V[5]], [V[0], V[1], V[5], V[4]], [V[0], V[4], V[5], V[3]]]
    return [[V[0], V[1], V[2], V[4]], [V[0], V[4], V[2], V[5]], [V[0], V[4], V[5], V[3]]]


# ------------------------------------------------------------------------------------------------ space
@dataclass
class EdgeSpace:
    """Edges of a simplex mesh and the element data needed by the Whitney basis."""

    points: np.ndarray
    cells: np.ndarray             # (M, d+1), vertices sorted ascending in each cell
    edges: np.ndarray             # (E, 2) global edges, i < j
    cell_edges: np.ndarray        # (M, n_local) global edge index of each local edge
    grads: np.ndarray             # (M, d+1, d) gradients of the barycentric coordinates
    vol: np.ndarray               # (M,)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def local_edges(self):
        return _TRI_EDGES if self.dim == 2 else _TET_EDGES

    @classmethod
    def of(cls, points, cells) -> EdgeSpace:
        pts = np.asarray(points, float)
        cl = np.sort(np.asarray(cells, np.int64), axis=1)
        d = pts.shape[1]
        if cl.shape[1] != d + 1:
            raise ValueError("cells must be triangles in 2-D or tetrahedra in 3-D")
        loc = _TRI_EDGES if d == 2 else _TET_EDGES
        pairs = np.stack([cl[:, [i, j]] for i, j in loc], axis=1)          # (M, nl, 2), ascending
        flat = pairs.reshape(-1, 2)
        edges, inv = np.unique(flat, axis=0, return_inverse=True)
        cell_edges = inv.reshape(len(cl), len(loc))
        P = pts[cl]                                                        # (M, d+1, d)
        A = np.concatenate([np.ones(P.shape[:2] + (1,)), P], axis=2)       # rows [1, x, y(, z)]
        Ainv = np.linalg.inv(A)                                            # columns: coefficients of l_i
        grads = np.transpose(Ainv[:, 1:, :], (0, 2, 1))                    # (M, d+1, d)
        vol = np.abs(np.linalg.det(A)) / math.factorial(d)
        if np.any(vol <= 0):
            raise ValueError("degenerate element")
        return cls(pts, cl, edges, cell_edges, grads, vol)

    # -- basis on reference data
    def curl_local(self) -> np.ndarray:
        """Curl of each local basis function, constant per element: (M, nl) in 2-D, (M, nl, 3) in 3-D."""
        G = self.grads
        if self.dim == 2:
            return np.stack([2 * (G[:, i, 0] * G[:, j, 1] - G[:, i, 1] * G[:, j, 0]) for i, j in self.local_edges], 1)
        return np.stack([2 * np.cross(G[:, i], G[:, j]) for i, j in self.local_edges], 1)

    def basis_at(self, cell: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Values of the local basis functions at barycentric coordinates ``lam`` (P, d+1) in elements ``cell``
        (P,): (P, nl, d)."""
        G = self.grads[cell]
        return np.stack([lam[:, i, None] * G[:, j] - lam[:, j, None] * G[:, i] for i, j in self.local_edges], 1)

    def locate(self, pts: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """Containing element and barycentric coordinates of each point: candidates are the elements with the
        nearest centroids, widened until every point is found."""
        from scipy.spatial import cKDTree

        pts = np.atleast_2d(np.asarray(pts, float))
        cent = self.points[self.cells].mean(1)
        tree = cKDTree(cent)
        cell = np.full(len(pts), -1)
        lam = np.zeros((len(pts), self.dim + 1))
        kk = min(k, len(cent))
        while True:
            todo = np.flatnonzero(cell < 0)
            _, cand = tree.query(pts[todo], k=kk)
            cand = cand.reshape(len(todo), -1)
            for c in range(cand.shape[1]):
                still = cell[todo] < 0
                if not np.any(still):
                    break
                e = cand[still, c]
                idx = todo[still]
                G = self.grads[e]
                p0 = self.points[self.cells[e, 0]]
                l = np.einsum("pkd,pd->pk", G, pts[idx] - p0)
                l[:, 0] += 1.0 - l.sum(1)
                ok = np.all(l >= -1e-10, axis=1)
                cell[idx[ok]] = e[ok]
                lam[idx[ok]] = l[ok]
            if np.all(cell >= 0) or kk >= len(cent):
                break
            kk = min(4 * kk, len(cent))
        if np.any(cell < 0):
            raise ValueError(f"{int(np.sum(cell < 0))} points are outside the mesh")
        return cell, lam

    def evaluate(self, x: np.ndarray, pts) -> np.ndarray:
        """Field ``E`` at points from edge coefficients ``x``: (P, d)."""
        cell, lam = self.locate(pts)
        N = self.basis_at(cell, lam)
        coef = x[self.cell_edges[cell]]
        return np.einsum("pk,pkd->pd", coef, N)

    def curl_at(self, x: np.ndarray, pts) -> np.ndarray:
        cell, _ = self.locate(pts)
        C = self.curl_local()[cell]
        coef = x[self.cell_edges[cell]]
        return np.einsum("pk,pk...->p...", coef, C)


def _as_tensor(m, M: int) -> np.ndarray:
    """Per-element 3x3 tensors from a scalar, (M,), (3, 3), (M, 3) diagonal or (M, 3, 3)."""
    a = np.asarray(m)
    if a.ndim == 0:
        return np.broadcast_to(a * np.eye(3), (M, 3, 3))
    if a.shape == (M,):
        return a[:, None, None] * np.eye(3)
    if a.shape == (3, 3):
        return np.broadcast_to(a, (M, 3, 3))
    if a.shape == (M, 3):
        return a[:, :, None] * np.eye(3)
    if a.shape == (M, 3, 3):
        return a
    raise ValueError(f"material shape {a.shape} not understood for {M} elements")


def assemble(space: EdgeSpace, eps_r=1.0, mu_r=1.0) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Curl-curl ``S_ab = int (mu_r^-1 curl N_a) . curl N_b`` and mass ``T_ab = int (eps_r N_a) . N_b``."""
    M = len(space.cells)
    d = space.dim
    eps = _as_tensor(eps_r, M)
    nu = np.linalg.inv(_as_tensor(mu_r, M))
    C = space.curl_local()
    if d == 2:
        Se = space.vol[:, None, None] * nu[:, 2, 2][:, None, None] * C[:, :, None] * C[:, None, :]
    else:
        Se = space.vol[:, None, None] * np.einsum("mai,mij,mbj->mab", C, nu, C)
    G = space.grads
    epsd = eps[:, :d, :d]
    Q = np.einsum("mid,mde,mje->mij", G, epsd, G)                        # grad l_i . eps grad l_j
    n = d + 1
    Mll = space.vol[:, None, None] * (np.ones((n, n)) + np.eye(n)) / ((d + 1) * (d + 2))
    loc = space.local_edges
    nl = len(loc)
    Te = np.zeros((M, nl, nl), complex if np.iscomplexobj(Q) else float)
    for a, (i, j) in enumerate(loc):
        for b, (k, l) in enumerate(loc):
            Te[:, a, b] = (Mll[:, i, k] * Q[:, j, l] - Mll[:, i, l] * Q[:, j, k]
                           - Mll[:, j, k] * Q[:, i, l] + Mll[:, j, l] * Q[:, i, k])
    rows = np.repeat(space.cell_edges, nl, axis=1).ravel()
    cols = np.tile(space.cell_edges, (1, nl)).ravel()
    E = space.n_edges
    S = sp.csr_matrix((Se.ravel(), (rows, cols)), shape=(E, E))
    T = sp.csr_matrix((Te.ravel(), (rows, cols)), shape=(E, E))
    return S, T


def boundary_edges(space: EdgeSpace) -> np.ndarray:
    """Edges on the outer boundary (edges of boundary facets: facets that belong to one element only)."""
    cl = space.cells
    d = space.dim
    if d == 2:
        faces = np.concatenate([cl[:, [0, 1]], cl[:, [0, 2]], cl[:, [1, 2]]])
    else:
        faces = np.concatenate([cl[:, [0, 1, 2]], cl[:, [0, 1, 3]], cl[:, [0, 2, 3]], cl[:, [1, 2, 3]]])
    faces = np.sort(faces, axis=1)
    uniq, counts = np.unique(faces, axis=0, return_counts=True)
    bf = uniq[counts == 1]
    if d == 2:
        pairs = bf
    else:
        pairs = np.concatenate([bf[:, [0, 1]], bf[:, [0, 2]], bf[:, [1, 2]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    # map to global edge indices
    key = space.edges[:, 0] * (len(space.points) + 1) + space.edges[:, 1]
    kp = pairs[:, 0] * (len(space.points) + 1) + pairs[:, 1]
    order = np.argsort(key)
    pos = np.searchsorted(key[order], kp)
    return np.sort(order[pos])


def _free(space: EdgeSpace, pec: np.ndarray | None) -> np.ndarray:
    pec = boundary_edges(space) if pec is None else np.asarray(pec, np.int64)
    free = np.ones(space.n_edges, bool)
    free[pec] = False
    return np.flatnonzero(free)


def gradient_matrix(space: EdgeSpace) -> sp.csr_matrix:
    """Edge-node incidence ``G`` (``E x N``): the gradient of nodal hat function ``k`` in the edge basis is
    ``sum_e G[e, k] N_e`` with ``G[e, k] = +1`` if ``k`` is the edge's end node, ``-1`` if its start node (Whitney's
    property: gradients lie exactly in the edge space)."""
    E = space.n_edges
    rows = np.repeat(np.arange(E), 2)
    cols = space.edges.ravel()
    vals = np.tile([-1.0, 1.0], E)
    return sp.csr_matrix((vals, (rows, cols)), shape=(E, len(space.points)))


def eigenmodes(space: EdgeSpace, n: int = 6, eps_r=1.0, mu_r=1.0, pec: np.ndarray | None = None,
               k_guess: float | None = None, return_vectors: bool = False):
    """Lowest ``n`` non-zero eigenvalues ``k^2`` of ``S x = k^2 T x`` with PEC on ``pec`` edges (default: the whole
    boundary), with their multiplicity.

    The curl-curl operator has a large null space (gradients of nodal functions, ``k = 0``). It is removed with the
    discrete divergence constraint ``G^T T x = 0`` (``G`` the edge-node incidence) in mixed form (F. Kikuchi, Comput.
    Methods Appl. Mech. Eng. 64 (1987) 509):

        [S - s T   T G] [x]   [T x]
        [G^T T      0 ] [p] = [ 0 ] / (k^2 - s)

    Shift-invert at ``s`` maps the constraint directions to eigenvalue 0 of the operator, where ARPACK ignores them,
    so the ``n`` eigenvalues nearest ``s`` are physical. ``s = k_guess^2``, by default ``(0.9 pi / L)^2`` (``L`` the
    largest extent of the mesh), just below the lowest resonance of a closed PEC domain: with ``s`` below the lowest
    mode the ``n`` nearest are the ``n`` lowest."""
    S, T = assemble(space, eps_r, mu_r)
    fr = _free(space, pec)
    if k_guess is None:
        k_guess = 0.9 * math.pi / np.ptp(space.points, axis=0).max()
    vals, vecs = constrained_eigs(S, T, gradient_matrix(space), space.edges, fr, n, k_guess ** 2)
    if not return_vectors:
        return vals
    full = np.zeros((space.n_edges, len(vals)), complex)
    full[fr] = vecs
    return vals, full


def constrained_eigs(S, T, G, edges: np.ndarray, free: np.ndarray, n: int, shift: float):
    """The mixed (divergence-constrained) shift-invert eigensolver of :func:`eigenmodes` for any edge-element
    matrices: ``S``, ``T`` (``E x E``), the edge-node incidence ``G`` (``E x N``), the edges' node pairs (to find the
    nodes on constrained edges), the free edge indices and the shift. Returns the ``n`` eigenvalues nearest the
    shift, sorted, and their vectors on the free edges."""
    fixed = np.setdiff1d(np.arange(S.shape[0]), free)
    bnd_nodes = np.unique(edges[fixed].ravel()) if len(fixed) else np.zeros(0, np.int64)
    nodes = np.setdiff1d(np.arange(G.shape[1]), bnd_nodes)
    G = G.tocsr()[free][:, nodes]
    S, T = S.tocsr()[free][:, free].tocsc(), T.tocsr()[free][:, free].tocsc()
    TG = (T @ G).tocsc()
    K = sp.bmat([[S - shift * T, TG], [TG.T, None]], format="csc")
    cplx = np.iscomplexobj(K.data)
    lu = spla.splu(K.astype(complex if cplx else float))
    ne, nn = S.shape[0], G.shape[1]

    def op(v):
        return lu.solve(np.concatenate([T @ v, np.zeros(nn, dtype=np.result_type(v, T.dtype))]))[:ne]

    A = spla.LinearOperator((ne, ne), matvec=op, dtype=complex if cplx else float)
    nu, vecs = spla.eigs(A, k=min(n, ne - 2), which="LM", ncv=min(ne - 1, max(4 * n + 10, 40)))
    lam = shift + 1.0 / nu
    order = np.argsort(lam.real)
    vals = lam[order]
    if np.allclose(vals.imag, 0, atol=1e-9 * np.abs(vals).max()):
        vals = vals.real
    return vals, vecs[:, order]


def edge_averages(space: EdgeSpace, x: np.ndarray):
    """``(midpoints, unit tangents, E . t averaged along each edge)``: the degrees of freedom divided by the edge
    lengths. Edge averages are the most accurate point data of the lowest-order space (second order for smooth
    fields, where point values inside an element are first order)."""
    p0 = space.points[space.edges[:, 0]]
    p1 = space.points[space.edges[:, 1]]
    t = p1 - p0
    L = np.linalg.norm(t, axis=1)
    return 0.5 * (p0 + p1), t / L[:, None], x / L


# quadrature on the reference simplex: barycentric points and weights (sum of weights = 1)
def _quadrature(d: int):
    if d == 2:          # degree 4, 6 points (Dunavant 1985)
        a, b = 0.445948490915965, 0.091576213509771
        w1, w2 = 0.223381589678011, 0.109951743655322
        pts = [(a, a, 1 - 2 * a), (a, 1 - 2 * a, a), (1 - 2 * a, a, a), (b, b, 1 - 2 * b), (b, 1 - 2 * b, b),
               (1 - 2 * b, b, b)]
        return np.array(pts), np.array([w1] * 3 + [w2] * 3)
    # degree 3, 5 points (Keast); the negative weight is exact
    pts = [(0.25, 0.25, 0.25, 0.25)] + [tuple(0.5 if i == k else 1 / 6 for i in range(4)) for k in range(4)]
    return np.array(pts), np.array([-0.8] + [0.45] * 4)


def _quad_points(space: EdgeSpace):
    lam, w = _quadrature(space.dim)
    P = space.points[space.cells]                                        # (M, d+1, d)
    X = np.einsum("qk,mkd->mqd", lam, P)                                # (M, Q, d)
    return lam, w, X


def load_from_function(space: EdgeSpace, f) -> np.ndarray:
    """``b_a = int f . N_a`` for a vector function ``f(X) -> (n, d)`` (complex allowed), by a degree-4 (triangles)
    or degree-3 (tetrahedra) rule on every element."""
    lam, w, X = _quad_points(space)
    M, Q, d = X.shape
    F = np.asarray(f(X.reshape(-1, d))).reshape(M, Q, d)
    loc = space.local_edges
    G = space.grads
    out = np.zeros(space.n_edges, complex if np.iscomplexobj(F) else float)
    for a, (i, j) in enumerate(loc):
        N = lam[None, :, i, None] * G[:, None, j] - lam[None, :, j, None] * G[:, None, i]   # (M, Q, d)
        val = space.vol * np.einsum("q,mqd,mqd->m", w, N, F)
        np.add.at(out, space.cell_edges[:, a], val)
    return out


def edge_dofs(space: EdgeSpace, E, edges: np.ndarray | None = None, order: int = 4) -> np.ndarray:
    """Degrees of freedom of a field: the line integrals ``int_e E . t dl`` along each edge (start node to end
    node), by Gauss-Legendre with ``order`` points. ``edges`` restricts them (e.g. to boundary edges for an
    inhomogeneous condition)."""
    edges = np.arange(space.n_edges) if edges is None else np.asarray(edges)
    s, w = np.polynomial.legendre.leggauss(order)
    p0 = space.points[space.edges[edges, 0]]
    p1 = space.points[space.edges[edges, 1]]
    tvec = p1 - p0
    out = 0.0
    for sk, wk in zip(s, w, strict=True):
        X = p0 + 0.5 * (sk + 1) * tvec
        out = out + 0.5 * wk * np.einsum("ed,ed->e", np.asarray(E(X)), tvec)
    return np.asarray(out)


def errors(space: EdgeSpace, x: np.ndarray, E, curlE=None) -> tuple[float, float]:
    """Relative L2 errors of E and of curl E against exact functions (same quadrature as the loads)."""
    lam, w, X = _quad_points(space)
    M, Q, d = X.shape
    Eh = np.zeros((M, Q, d), complex)
    G = space.grads
    for a, (i, j) in enumerate(space.local_edges):
        N = lam[None, :, i, None] * G[:, None, j] - lam[None, :, j, None] * G[:, None, i]
        Eh += x[space.cell_edges[:, a]][:, None, None] * N
    Ee = np.asarray(E(X.reshape(-1, d))).reshape(M, Q, d)
    num = np.einsum("m,q,mqd->", space.vol, w, np.abs(Eh - Ee) ** 2)
    den = np.einsum("m,q,mqd->", space.vol, w, np.abs(Ee) ** 2)
    eE = float(np.sqrt(num / den))
    if curlE is None:
        return eE, float("nan")
    C = space.curl_local()
    Ch = np.einsum("mk,mk...->m...", x[space.cell_edges], C)
    Ce = np.asarray(curlE(X.reshape(-1, d))).reshape((M, Q) + (() if d == 2 else (3,)))
    if d == 2:
        num = np.einsum("m,q,mq->", space.vol, w, np.abs(Ch[:, None] - Ce) ** 2)
        den = np.einsum("m,q,mq->", space.vol, w, np.abs(Ce) ** 2)
    else:
        num = np.einsum("m,q,mqc->", space.vol, w, np.abs(Ch[:, None] - Ce) ** 2)
        den = np.einsum("m,q,mqc->", space.vol, w, np.abs(Ce) ** 2)
    return eE, float(np.sqrt(num / den))


def dipole_load(space: EdgeSpace, point, moment, omega: float) -> np.ndarray:
    """Load vector ``b_a = -j omega mu0 (I l) . N_a(r0)`` of a point current ``I l`` (A m, vector) at ``point``."""
    cell, lam = space.locate(np.atleast_2d(point))
    N = space.basis_at(cell, lam)[0]                                     # (nl, d)
    b = np.zeros(space.n_edges, complex)
    np.add.at(b, space.cell_edges[cell[0]], -1j * omega * MU0 * (N @ np.asarray(moment)[:space.dim]))
    return b


def solve_driven(space: EdgeSpace, omega: float, load: np.ndarray, eps_r=1.0, mu_r=1.0,
                 pec: np.ndarray | None = None, pec_values: np.ndarray | None = None) -> np.ndarray:
    """Edge coefficients of E for ``(S - k0^2 T) x = load`` (sparse direct solve). The ``pec`` edges (default: the
    whole boundary) take ``pec_values`` (default 0; e.g. ``-edge_dofs(space, E_inc, pec)`` on a conductor in a
    scattered-field formulation)."""
    k0 = omega / C0
    S, T = assemble(space, eps_r, mu_r)
    A = (S - k0 * k0 * T).tocsr().astype(complex)
    fr = _free(space, pec)
    fixed = np.setdiff1d(np.arange(space.n_edges), fr)
    x = np.zeros(space.n_edges, complex)
    if pec_values is not None:
        x[fixed] = pec_values
    rhs = np.asarray(load, complex)[fr] - A[fr][:, fixed] @ x[fixed]
    x[fr] = spla.spsolve(A[fr][:, fr].tocsc(), rhs)
    return x


def pml_tensors(centroids: np.ndarray, inner: Sequence[tuple[float, float]], thickness: float, omega: float,
                sigma_max: float | None = None, order: int = 2, eps_r=1.0, mu_r=1.0):
    """Stretched-coordinate PML as anisotropic materials (Jin ch. 9; ``exp(+j omega t)``): outside the box ``inner``
    each axis gets ``s = 1 - j sigma(d) / (omega eps0)`` with ``sigma = sigma_max (d / thickness)^order``, and the
    materials become ``eps_r Lambda``, ``mu_r Lambda`` with ``Lambda = diag(s_y s_z / s_x, s_x s_z / s_y,
    s_x s_y / s_z)``. ``sigma_max`` defaults to a theoretical normal-incidence reflection of 1e-6. Returns
    per-element (M, 3, 3) tensors ``(eps, mu)``."""
    c = np.asarray(centroids, float)
    M, d = c.shape
    if sigma_max is None:
        sigma_max = -(order + 1) * math.log(1e-6) * EPS0 * C0 / (2 * thickness)
    s = np.ones((M, 3), complex)
    for ax in range(d):
        lo, hi = inner[ax]
        depth = np.maximum(lo - c[:, ax], 0) + np.maximum(c[:, ax] - hi, 0)
        sig = sigma_max * np.clip(depth / thickness, 0, 1) ** order
        s[:, ax] = 1 - 1j * sig / (omega * EPS0)
    lam = np.stack([s[:, 1] * s[:, 2] / s[:, 0], s[:, 0] * s[:, 2] / s[:, 1], s[:, 0] * s[:, 1] / s[:, 2]], 1)
    L = lam[:, :, None] * np.eye(3)
    return _as_tensor(eps_r, M) @ L, _as_tensor(mu_r, M) @ L


def waveguide_modes(points, tris, n: int = 6, k_guess: float | None = None) -> list[tuple[float, str]]:
    """Cut-off wavenumbers ``k_c`` (rad/m) of the lowest ``n`` modes of a hollow PEC waveguide with the given
    cross-section mesh, as ``(k_c, "TE" | "TM")`` sorted. TE: the transverse E field on edge elements with PEC
    (``curl curl E_t = k_c^2 E_t``); TM: ``E_z`` on P1 nodes with ``E_z = 0`` on the wall (``-lap E_z = k_c^2 E_z``).
    For a filled guide divide ``k_c`` by ``sqrt(eps_r mu_r)`` to get the cut-off of the filling."""
    pts = np.asarray(points, float)
    space = EdgeSpace.of(pts, tris)
    te = eigenmodes(space, n, k_guess=k_guess)
    # TM: nodal stiffness and consistent mass on P1 triangles
    from .fem import Geometry
    from .mesh import Mesh2D, boundary_nodes

    t = np.asarray(tris, np.int64)
    mesh = Mesh2D(pts, _ccw(pts, t), np.zeros(len(t), np.int64), ["guide"])
    g = Geometry.of(mesh)
    K = g.stiffness(np.ones(len(t)))
    me = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    Me = g.area[:, None, None] * me[None]
    Mm = sp.csr_matrix((Me.ravel(), (g.rows, g.cols)), shape=(g.n, g.n))
    inner = np.setdiff1d(np.arange(len(pts)), boundary_nodes(mesh.tris))
    Ki, Mi = K[inner][:, inner].tocsc(), Mm[inner][:, inner].tocsc()
    s = (k_guess if k_guess is not None else 0.9 * math.pi / np.ptp(pts, axis=0).max()) ** 2
    tm = np.sort(spla.eigsh(Ki, k=min(n, Ki.shape[0] - 2), M=Mi, sigma=s, which="LM")[0])
    modes = [(float(math.sqrt(v)), "TE") for v in te] + [(float(math.sqrt(v)), "TM") for v in tm]
    return sorted(modes)[:n]


def _ccw(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = points[tris]
    det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    t = tris.copy()
    t[det < 0, 1], t[det < 0, 2] = tris[det < 0, 2], tris[det < 0, 1]
    return t


__all__ += ["waveguide_modes"]
