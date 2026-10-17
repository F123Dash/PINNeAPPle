"""Surrogate Studio app (apps/surrogate_studio): HTTP smoke test of the routes, an example and an upload."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sklearn")
pytest.importorskip("pinneapple_core.service")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "apps" / "surrogate_studio" / "examples" / "airfoil_rans_runs.csv"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SUR_DATA_DIR"] = str(tmp_path_factory.mktemp("sur"))
    sys.path.insert(0, str(ROOT / "apps" / "surrogate_studio"))
    from surrogate_studio.api import app
    return TestClient(app)


def _wait(c, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise TimeoutError(job_id)


def test_health_meta_page(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert {e["name"] for e in m["examples"]} == {"airfoil_rans", "burgers_fields"}
    assert {x["id"] for x in m["methods"]} >= {"gp", "mlp", "fno", "deeponet", "pinn"}
    assert "Surrogate Studio" in client.get("/").text and client.get("/docs").status_code == 200


def test_example_study_hosts_a_model(client):
    job = _wait(client, client.post("/api/studies", json={"example": "airfoil_rans"}).json()["id"])
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert r["study"]["best"] and r["study"]["trust"]["decision"] in ("APPROVED", "REVIEW", "REJECT")
    assert r["model_ref"]["kind"] == "ModelRef" and r["model_ref"]["metrics_split"] == "test"
    assert {"study_result.json", "model_ref.json", "dataset_spec.json", "report.md", "model.zip"} <= set(r["files"])
    assert client.get(f"/api/studies/{job['id']}/files/report.md").text.startswith("# Surrogate study")
    name = r["model"]["name"]
    p = client.post(f"/v1/models/{name}/predict", json={"inputs": {"T0": 0.15, "T1": 0.12, "T2": 0.13, "M0": 0.02,
                                                                   "M1": 0.05, "M2": 0.04, "alpha_deg": [4, 30]}})
    assert p.status_code == 200 and not p.json()["ood"][0]["ood"] and p.json()["ood"][1]["ood"]
    assert client.get("/v1/metrics").json()["calls"] >= 1
    assert client.get("/api/usage").json()["units"]["inferences"]["used"] >= 2


def test_upload_inspect_and_train(client):
    with open(CSV, "rb") as f:
        info = client.post("/api/inspect", files={"file": ("runs.csv", f, "text/csv")}).json()
    assert info["n_rows"] == 374 and info["suggested"]["group"] == "shape_id"
    body = {"upload_id": info["upload_id"], "inputs": ["T0", "T1", "T2", "M0", "M1", "M2", "alpha_deg"],
            "outputs": ["Cl"], "group": "shape_id", "methods": ["ridge", "gbt"], "budget_s": 0.5,
            "model_name": "upload test", "ood_policy": "refuse", "constraints": ["Cl > -2"]}
    job = _wait(client, client.post("/api/studies", json=body).json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["result"]["model"]["name"] == "upload-test"
    p = client.post("/v1/models/upload-test/predict", json={"inputs": [{"T0": 0.15, "T1": 0.12, "T2": 0.13,
                    "M0": 0.02, "M1": 0.05, "M2": 0.04, "alpha_deg": 40}]}).json()
    assert p["outputs"]["Cl"] == [None] and p["ood_policy"] == "refuse"
    bad = client.post("/api/studies", json={**body, "outputs": ["nope"]})
    assert bad.status_code == 422
    assert client.post("/api/studies", json={**body, "budget_s": 1e6}).status_code == 422
