"""Engineering GUI Agent app (apps/guiagent): the HTTP API around pinneapple_tools.automation.

The applications (FreeCAD, OpenFOAM, ParaView) are replaced by an empty toolchain, so a run is recorded as blocked
at its first step; the full chain is tested in tests/test_automation.py where the applications are installed."""
from __future__ import annotations

import importlib
import io
import sys
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIAGENT_DATA", str(tmp_path))
    monkeypatch.delenv("GUIAGENT_USER", raising=False)
    monkeypatch.delenv("GUIAGENT_MODE", raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "apps" / "guiagent"))
    sys.modules.pop("guiagent.api", None)
    api = importlib.import_module("guiagent.api")
    from pinneapple_tools.automation import Toolchain
    api._toolchain = Toolchain()                                 # no applications: steps are blocked, not run
    return TestClient(api.app), api


def _wait(c, run_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/runs/{run_id}").json()
        if not r["active"] and r["state"] not in ("running", "pending"):
            return r
        time.sleep(0.1)
    raise AssertionError("run did not stop")


def test_pages_workflows_and_interpretation(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert "Engineering GUI Agent" in c.get("/").text
    assert c.get("/shared/app.css").status_code == 200
    wfs = c.get("/api/workflows").json()
    wf = next(w for w in wfs if w["id"] == "channel_obstacle")
    assert [p["action"] for p in wf["plan"]][:2] == ["freecad.open", "freecad.macro"]
    assert wf["inputs"]["obstacle_height_mm"]["max"] == 80
    it = c.post("/api/interpret", json={"task": "channel with a 6 cm obstacle, inlet 25 mm/s"}).json()
    assert it["workflow"] == "channel_obstacle" and it["resolved_inputs"] == {"obstacle_height_mm": 60.0,
                                                                             "inlet_velocity_mm_s": 25.0}
    assert c.post("/api/interpret", json={"task": "obstacle height 300 mm in the channel"}).status_code == 422
    meta = c.get("/api/meta").json()
    assert meta["mode"] == "public" and meta["uploads"] is False


def test_run_is_recorded_and_unsafe_requests_refused(client):
    c, _ = client
    assert c.post("/api/runs", json={"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 99}}).status_code == 422
    assert c.post("/api/runs", json={"workflow": "nope"}).status_code == 404
    r = c.post("/api/runs", json={"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 30}})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    rec = _wait(c, run_id)
    assert rec["state"] == "blocked" and rec["steps"][0]["state"] == "blocked"
    assert "missing" in rec["steps"][0]["error"] and "folder" not in rec
    assert c.get(f"/api/runs/{run_id}/report").status_code == 200
    sr = c.get(f"/api/runs/{run_id}/study_result").json()
    assert sr["schema"] == "pinneapple.study_result/0" and sr["decision"] == "INCOMPLETE"
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/runs/{run_id}/bundle").content))
    assert f"{run_id}/run.json" in z.namelist()
    assert c.get(f"/api/runs/{run_id}/files/../../etc/passwd").status_code == 404
    assert c.get("/api/runs/..%2F..%2Fetc/files/x").status_code == 404
    assert c.post(f"/api/runs/{run_id}/approve", json={"step": "solve"}).status_code == 409
    assert [x["run_id"] for x in c.get("/api/runs").json()] == [run_id]
    up = c.post("/api/workflows", files={"file": ("w.zip", b"PK", "application/zip")})
    assert up.status_code == 403                                 # public demo: no uploads, no macros


def test_one_job_at_a_time(client):
    c, api = client
    assert api._jobs.acquire(blocking=False)                    # someone else's run holds the only slot
    try:
        r = c.post("/api/runs", json={"workflow": "channel_obstacle"})
        assert r.status_code == 429
        runs = list(Path(api.RUNS).glob("*/run.json")) if Path(api.RUNS).is_dir() else []
        assert runs == []                                        # no orphan run folder
    finally:
        api._jobs.release()
