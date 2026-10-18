"""Lab seal and "how to raise it" hooks: the letter is read from pinneapple_lab (curation tier today, the A-F seal
scale and curation.next_steps when the Lab provides them), never graded by the flow library."""
import json

import pytest

import pinneapple_flow as pf
from pinneapple_flow import lab_status as ls
from pinneapple_flow.decisions import RAISE_NODES, rule_decision, user_decision
from pinneapple_flow.service import FLOWS_DIR

ITEM = "flow_demo"


def _curation(tmp_path, **rec):
    record = {"tier": "C", "tier_name": "exploratory", "gaps": ["compare with an independent reference"],
              "headline": {"text": "PASS rel_l2 vs exact"}, **rec}
    (tmp_path / "curation.json").write_text(json.dumps({"generated": "now", "items": {ITEM: record}}))
    return str(tmp_path)


def test_seal_status_reads_the_curation_tier(tmp_path):
    lab = _curation(tmp_path)
    st = ls.seal_status(ITEM, lab)
    assert st["letter"] == "C" and st["name"] == "exploratory" and st["reason"] == "PASS rel_l2 vs exact"
    assert not st["sealed"]                      # today's seal API seals A and B only
    assert ls.seal_status("unknown", lab)["letter"] is None
    assert ls.seal_status(ITEM, str(tmp_path / "empty"))["reason"].startswith("not curated")


def test_seal_status_follows_a_full_scale_when_the_lab_has_it(tmp_path, monkeypatch):
    from pinneapple_lab import curation
    monkeypatch.setattr(curation, "SEALS", {k: k.lower() for k in "ABCDEF"}, raising=False)
    lab = _curation(tmp_path, seal="E", seal_name="known problems", failing_check="mass balance 3% > 1%",
                    how_to_raise="refine the mesh")
    st = ls.seal_status(ITEM, lab)
    assert (st["letter"], st["scale"], st["failing_check"], st["hint"]) == ("E", "ABCDEF", "mass balance 3% > 1%",
                                                                           "refine the mesh")


def test_next_steps_degrades_then_maps_lab_actions_to_registered_nodes(tmp_path, monkeypatch):
    from pinneapple_lab import curation
    lab = _curation(tmp_path)
    monkeypatch.delattr(curation, "next_steps", raising=False)
    out = ls.next_steps(ITEM, lab)
    assert not out["available"] and out["message"] == ls.NOT_AVAILABLE
    assert out["gaps"][0]["gap"].startswith("compare") and out["gaps"][0]["nodes"] == []

    def fake(item, lab=None):
        return [{"gap": "no independent reference", "action": "validate.compare", "command": "", "params": {}},
                {"gap": "no uncertainty", "action": "uq", "command": "pinneapple_analysis.sensitivity:sobol_indices",
                 "params": {"n": 64, "bogus": 1}},
                {"gap": "nothing registered", "action": "write a paper", "command": "", "params": {}}]
    monkeypatch.setattr(curation, "next_steps", fake, raising=False)
    out = ls.next_steps(ITEM, lab)
    assert out["available"]
    assert out["gaps"][0]["nodes"] == [{"type": "validate.compare", "params": {}}]
    assert out["gaps"][1]["nodes"][0]["type"] == "lib.uq.sobol"
    assert "bogus" not in out["gaps"][1]["nodes"][0]["params"]
    assert out["gaps"][2]["nodes"] == []


def test_insert_nodes_wires_by_port_type():
    doc = json.loads((FLOWS_DIR / "heat_1d.json").read_text())
    new, missing = ls.insert_nodes(doc, "pinn", [{"type": "validate.compare"}])
    assert not missing
    into = {e["target_port"]: e["source"] for e in new["edges"] if e["target"] == "compare"}
    assert into["candidates"] == "pinn" and into["reference"] != "pinn"
    assert not [i for i in pf.validate_flow(new) if i.level == "error"]
    with pytest.raises(KeyError):
        ls.insert_nodes(doc, "pinn", [{"type": "os.system"}])


def test_weak_decisions_suggest_registered_nodes():
    for topic, types in RAISE_NODES.items():
        assert types and all(t in pf.REGISTRY for t in types), topic
    weak = user_decision("n", "method", "picked by hand")
    assert weak.no_reference and weak.raise_with == list(RAISE_NODES["method"])
    strong = rule_decision("family.heat", what="heat")
    assert strong.raise_with == []


def test_api_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pinneapple_flow.studio import load_app
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("PINNEAPPLE_FLOW_DB_URL", raising=False)
    monkeypatch.delenv("FLOW_USER", raising=False)
    app = load_app()
    c = TestClient(app)
    assert c.get("/api/lab/seal", params={"item": "x"}).json()["letter"] is None
    assert "available" in c.get("/api/lab/next-steps", params={"item": "x"}).json()
    doc = json.loads((FLOWS_DIR / "heat_1d.json").read_text())
    r = c.post("/api/flows/insert-nodes", json={"flow": doc, "anchor": "pinn", "nodes": [{"type": "uq.sensitivity"}]})
    assert r.status_code == 200 and r.json()["valid"]
    assert c.post("/api/flows/insert-nodes", json={"flow": doc, "nodes": [{"type": "nope"}]}).status_code == 400
    app.state.service.shutdown()
