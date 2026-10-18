"""Openly licensed real series, downloaded from a pinned commit and checked against a SHA-256.

* :func:`load_usd_eur_daily` -- daily U.S. dollars per euro, 1999 onwards, from the Federal Reserve H.10 release
  via FRED (series DEXUSEU), as packaged by the `datasets/exchange-rates` data package (licence ODC-PDDL 1.0,
  https://github.com/datasets/exchange-rates). Pinned to commit ``c023dd8b``; the file holds all currencies.
* :func:`load_sunspots_yearly` -- yearly mean sunspot numbers 1700-2008 from the NOAA National Geophysical Data
  Center, in the public domain, as shipped with statsmodels (``statsmodels.datasets.sunspots``); read from the
  installed statsmodels when present, else from statsmodels v0.14.4 on GitHub.

Downloads go to ``cache_dir`` (default ``$PINNEAPPLE_DATA_CACHE`` or ``~/.cache/pinneapple/datasets``) and are
re-used when the checksum matches; a mismatch raises instead of returning different data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["PublicSeries", "PINNED", "fetch_pinned", "load_usd_eur_daily", "load_sunspots_yearly", "log_returns"]

PINNED: dict[str, dict[str, str]] = {
    "exchange_rates_daily": {
        "url": "https://raw.githubusercontent.com/datasets/exchange-rates/"
               "c023dd8b8f879610641a7391c0394c3bbb42d0f7/data/daily.csv",
        "sha256": "68349ef41ecc0cf35fa00ba0d9d63ec93b45a5182c37d2bb9a368fcd654bfeaa",
        "license": "ODC-PDDL-1.0 (data: Federal Reserve Board H.10 via FRED, public domain)",
        "source": "https://fred.stlouisfed.org/series/DEXUSEU",
    },
    "sunspots_yearly": {
        "url": "https://raw.githubusercontent.com/statsmodels/statsmodels/v0.14.4/statsmodels/datasets/"
               "sunspots/sunspots.csv",
        "sha256": "f67889b1d9002cd5227f0e0ef54e35b419cdd85a31279adef6f73fb41e5c0a9b",
        "license": "public domain (NOAA NGDC, as distributed with statsmodels)",
        "source": "http://www.ngdc.noaa.gov/stp/solar/solarda3.html",
    },
}


@dataclass
class PublicSeries:
    name: str
    time: list[str]
    values: np.ndarray
    source: dict[str, str] = field(default_factory=dict)


def _cache_dir(cache_dir: str | None) -> Path:
    d = Path(cache_dir or os.environ.get("PINNEAPPLE_DATA_CACHE", Path.home() / ".cache" / "pinneapple" / "datasets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_pinned(key: str, cache_dir: str | None = None, *, local: str | None = None) -> bytes:
    """Bytes of the pinned file ``key`` (from ``local``, the cache, or the network), checksum verified."""
    spec = PINNED[key]
    if local is not None and Path(local).exists():
        data = Path(local).read_bytes()
        if _sha256(data) == spec["sha256"]:
            return data
    path = _cache_dir(cache_dir) / f"{key}.csv"
    if path.exists():
        data = path.read_bytes()
        if _sha256(data) == spec["sha256"]:
            return data
    with urllib.request.urlopen(spec["url"], timeout=120) as r:      # noqa: S310 -- pinned https URL
        data = r.read()
    got = _sha256(data)
    if got != spec["sha256"]:
        raise ValueError(f"{key}: checksum mismatch (expected {spec['sha256']}, got {got})")
    path.write_bytes(data)
    return data


def load_usd_eur_daily(cache_dir: str | None = None) -> PublicSeries:
    """U.S. dollars per euro on business days (days without a quote are dropped)."""
    data = fetch_pinned("exchange_rates_daily", cache_dir)
    t, v = [], []
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        if len(row) == 3 and row[1] == "Euro" and row[2] not in ("", "."):
            t.append(row[0])
            v.append(float(row[2]))
    return PublicSeries("usd_per_eur_daily", t, np.asarray(v), dict(PINNED["exchange_rates_daily"]))


def load_sunspots_yearly(cache_dir: str | None = None) -> PublicSeries:
    local = None
    try:
        import statsmodels.datasets.sunspots as _s
        local = str(Path(_s.__file__).with_name("sunspots.csv"))
    except ImportError:
        pass
    data = fetch_pinned("sunspots_yearly", cache_dir, local=local)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))[1:]
    return PublicSeries("sunspots_yearly", [str(int(float(r[0]))) for r in rows],
                        np.asarray([float(r[1]) for r in rows]), dict(PINNED["sunspots_yearly"]))


def log_returns(values) -> np.ndarray:
    v = np.asarray(values, float)
    return np.diff(np.log(v))
