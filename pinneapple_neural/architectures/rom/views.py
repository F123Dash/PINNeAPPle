"""A held-out snapshot on its body: the full-order field, the ROM prediction and ``|ROM - full-order|`` on the mesh,
with the deformed shape when the snapshots carry a displacement, the boundary-condition and load regions of the
dataset, the error hotspots and a short text on where the error concentrates.

    from pinneapple_neural.architectures.rom.views import reconstruction_view
    fv, info = reconstruction_view(snaps, model, index, reported={"study mean": 0.044})
    fv.glb(), fv.meta            # for /shared/fieldview3d.js
    info["errors"]["von_mises"]["summary"]

The snapshot needs a mesh (``SnapshotSet.mesh``: connectivity, optionally its own node coordinates). Every number is
computed from the stored full-order snapshot and the model's prediction at its inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["reconstruction_view"]


def _deformation_field(snaps, mesh_unit: str):
    """The displacement field (name, factor to the mesh's length unit): the first point vector field that is one."""
    from pinneapple_tools.visualization.studio.fieldview import displacement_factor
    units = snaps.meta.get("units") or {}
    for f in snaps.fields:
        fac = displacement_factor(f["name"], units.get(f["name"]), f["components"], mesh_unit) \
            if f["where"] == "point" else None
        if fac is not None:
            return f["name"], fac
    return None, None


def reconstruction_view(snaps, model, index: int, *, reported: Mapping[str, float] | None = None,
                        grow: float = 0.1, max_faces: int | None = None, n_hotspots: int = 3,
                        full_label: str = "") -> tuple[Any, dict[str, Any]]:
    """View of snapshot ``index`` of ``snaps`` against ``model`` (a ``ReducedModel``). Named node sets of the dataset
    become markers (kind ``bc`` / ``load`` / ``region``) and, grown by ``grow`` times the body size, the regions the
    error is compared over ("clamp zone"; the remaining nodes are the "rest of the body"). ``reported``: label -> relative error of the study to quote next to the
    pointwise numbers. Returns ``(FieldView, info)``; ``info["errors"][field]`` holds the error map of each field."""
    from pinneapple_analysis.validation.error_map import error_map, grow_region, pointwise_error
    from pinneapple_tools.visualization.studio.fieldview import VIEW_MAX_FACES, field_view
    mesh = snaps.mesh(index)
    if mesh is None:
        raise ValueError("these snapshots carry no mesh (connectivity): there is no body to show")
    q = {"times": [float(snaps.times[index])]} if model.mode == "time" else {"params": snaps.params[index:index + 1]}
    pred = model.predict(**q)[0]
    units = snaps.meta.get("units") or {}
    mesh_unit = units.get("points") or units.get("x") or "m"
    full = full_label or snaps.meta.get("full_order") or "Full-order"
    dname, dfac = _deformation_field(snaps, mesh_unit)
    deform = {}
    if dname:
        deform[full] = snaps.field_values(dname)[index] * dfac
        deform["ROM"] = snaps.field_values(dname, pred[None])[0] * dfac
    P = mesh.points
    size = float(np.ptp(P, axis=0).max()) or 1.0
    sets = snaps.meta.get("node_sets") or {}
    markers: list[dict[str, Any]] = []
    regions: dict[str, np.ndarray] = {}
    for name, s in sets.items():
        idx = np.asarray(s["nodes"], np.int64)
        markers.append({"name": name, "kind": s.get("kind", "region"), "points": P[idx], "detail": s.get("detail", "")})
        regions[f"{name} zone"] = grow_region(P, idx, grow * size)
    if regions:
        inside = np.unique(np.concatenate(list(regions.values())))
        rest = np.setdiff1d(np.arange(len(P)), inside)
        if len(rest):
            regions["rest of the body"] = rest
    fields, errors = [], {}
    for f in snaps.fields:
        if f["where"] != "point" or f["size"] != len(P) * f["components"]:
            continue
        n, unit = f["name"], units.get(f["name"], "")
        tv, pv = snaps.field_values(n)[index], snaps.field_values(n, pred[None])[0]
        em = error_map(P, pv, tv, regions=regions, unit=unit, n_hotspots=n_hotspots)
        errors[n] = em.to_dict(reported)
        mag = " magnitude" if f["components"] > 1 else ""
        d_full, d_rom = (full, "ROM") if dname else (None, None)
        names = [f"{n}:full", f"{n}:rom", f"{n}:error"]
        fields += [{"name": names[0], "values": tv, "label": f"{full} {n}{mag}", "unit": unit, "group": n,
                    "deform": d_full},
                   {"name": names[1], "values": pv, "label": f"ROM {n}{mag}", "unit": unit, "group": n,
                    "deform": d_rom},
                   {"name": names[2], "values": pointwise_error(pv, tv), "label": f"|ROM − {full}| {n}",
                    "unit": unit, "kind": "error", "deform": d_full,
                    "note": f"relative L2 {100 * em.rel_l2:.2f} % on this snapshot"}]
        for k, h in enumerate(em.hotspots):
            markers.append({"name": f"E{k + 1}", "kind": "hotspot", "points": P[h["indices"]], "centre": h["at"],
                            "fields": names, "detail": f"|error| up to {h['max']:.3g} {unit}: "
                                                       f"{100 * h['share']:.0f} % of the squared error"})
    if not fields:
        raise ValueError("no point field of these snapshots lives on the mesh nodes")
    fv = field_view(mesh, fields, deform=deform, markers=markers,
                    max_faces=VIEW_MAX_FACES if max_faces is None else max_faces,
                    title=f"snapshot {index}", length_unit=mesh_unit, notes=[f"{full}: the stored snapshot; ROM: {model.method}"
                                                      + (f" rank {model.rank}" if model.rank else "")])
    info = {"input": {k: np.asarray(v, float).ravel().tolist() for k, v in q.items()}, "errors": errors,
            "deformation": dname, "regions": {k: int(len(v)) for k, v in regions.items()}, "full_order": full}
    return fv, info
