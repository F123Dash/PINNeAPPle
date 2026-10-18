"""The curated reference registry (``references.yaml``) and the check every cited reference goes through.

A reference is *registry* when its key is in the curated file, *verified* when a DOI / URL given by someone else (an
LLM, a user) resolved over the network, and *unverified* otherwise. Nothing here invents a citation: an unknown key
stays an unknown key, shown as such.

    ref("raissi2019pinn")["doi"]                     # '10.1016/j.jcp.2018.10.045'
    check({"key": "lu2021deeponet"})["status"]       # 'registry'
    check({"title": "Some paper", "doi": "10.1/x"})  # 'unverified' (offline) or 'verified' / 'not found' (online)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path(__file__).with_name("references.yaml")
KINDS = ("article", "book", "conference", "preprint")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    import yaml
    data = yaml.safe_load(REGISTRY_FILE.read_text()) or {}
    return {k: {"key": k, **v} for k, v in data.items()}


def ref(key: str) -> dict[str, Any]:
    r = registry().get(key)
    if r is None:
        raise KeyError(f"{key!r} is not in the reference registry ({REGISTRY_FILE.name})")
    return dict(r)


def link(r: dict[str, Any]) -> str:
    """A clickable link for a reference: its DOI, else its URL, else ''."""
    if r.get("doi"):
        return f"https://doi.org/{r['doi']}"
    return r.get("url") or ""


def problems(entry: dict[str, Any]) -> list[str]:
    """What is wrong with a registry entry (empty when it is well formed)."""
    out = []
    for f in ("title", "authors", "year", "venue", "kind"):
        if not entry.get(f):
            out.append(f"missing {f}")
    if entry.get("kind") not in KINDS:
        out.append(f"kind must be one of {KINDS}")
    if not isinstance(entry.get("year"), int) or not 1600 < entry.get("year", 0) < 2100:
        out.append("year must be an integer")
    if not isinstance(entry.get("authors"), list):
        out.append("authors must be a list")
    if entry.get("doi") and not _DOI.match(str(entry["doi"])):
        out.append("doi is not of the form 10.xxxx/...")
    if entry.get("url") and not str(entry["url"]).startswith("https://"):
        out.append("url must be https")
    if entry.get("kind") != "book" and not (entry.get("doi") or entry.get("url")):
        out.append("an article / paper needs a doi or url")
    if not entry.get("topics"):
        out.append("missing topics")
    return out


def _online_ok() -> bool:
    return os.environ.get("PINNEAPPLE_FLOW_VERIFY_REFS", "").lower() in ("1", "true", "yes")


def _resolves(url: str, timeout: float = 6.0) -> bool | None:
    """True / False when the URL answered, None when it could not be checked."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pinneapple-flow-refcheck"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - https only, checked above
            return 200 <= r.status < 400
    except Exception as exc:  # noqa: BLE001 - network failure = cannot check
        code = getattr(exc, "code", None)
        if code in (404, 410):
            return False
        return None


def check(reference: dict[str, Any] | str, *, online: bool | None = None) -> dict[str, Any]:
    """A reference as stored on a decision, with ``status``: ``registry`` (curated key), ``verified`` (its DOI / URL
    resolved), ``not found`` (the DOI / URL did not resolve) or ``unverified`` (offline, or no DOI / URL)."""
    if isinstance(reference, str):
        reference = {"key": reference}
    r = dict(reference or {})
    key = r.get("key")
    if key and key in registry():
        out = {**ref(key), **{k: v for k, v in r.items() if k in ("section", "supports")}}
        out["status"] = "registry"
        out["link"] = link(out)
        return out
    if key:
        r["note"] = f"key {key!r} is not in the curated registry"
    doi, url = str(r.get("doi") or "").strip(), str(r.get("url") or "").strip()
    if doi and not _DOI.match(doi):
        r["doi"], doi = "", ""
        r["note"] = (r.get("note", "") + "; malformed DOI dropped").strip("; ")
    if url and not url.startswith("https://"):
        r["url"], url = "", ""
    target = f"https://doi.org/{doi}" if doi else url
    status = "unverified"
    if target and (online if online is not None else _online_ok()):
        ok = _resolves(target)
        status = "verified" if ok else ("not found" if ok is False else "unverified")
    r["status"] = status
    r["link"] = target
    return r


def cite(*keys: str, supports: str = "") -> list[dict[str, Any]]:
    """Registry references for ``keys`` (KeyError for an unknown key)."""
    out = []
    for k in keys:
        r = ref(k)
        r["status"] = "registry"
        r["link"] = link(r)
        if supports:
            r["supports"] = supports
        out.append(r)
    return out


def format_reference(r: dict[str, Any]) -> str:
    authors = r.get("authors") or []
    a = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    bits = [b for b in (a, f"({r['year']})" if r.get("year") else "", r.get("title"), r.get("venue")) if b]
    s = " ".join(bits[:2]) + (". " + ". ".join(bits[2:]) if len(bits) > 2 else "")
    if r.get("link"):
        s += f". {r['link']}"
    if r.get("status") and r["status"] != "registry":
        s += f" [{r['status']}]"
    return s
