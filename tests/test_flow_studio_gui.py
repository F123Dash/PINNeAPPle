"""Playwright smoke test of the Flow Studio GUI: palette drag and drop, import a flow, run it, live status, the
result panel of a node, the planner dialog, the architecture preview. Skipped when Playwright, a Chromium build or
the built frontend is missing (it never downloads a browser)."""
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pw = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "apps" / "flow_studio" / "frontend" / "dist" / "index.html"
CHROMIUM = [os.environ.get("PLAYWRIGHT_CHROMIUM", ""), "/opt/pw-browsers/chromium"]
SHOTS = os.environ.get("FLOW_SCREENSHOTS")          # a folder: save the screenshots there


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not DIST.is_file():
        pytest.skip("frontend not built (apps/flow_studio/frontend: npm run build)")
    os.environ["PINNEAPPLE_FLOW_HOME"] = str(tmp_path_factory.mktemp("flowhome"))
    import uvicorn

    from pinneapple_flow.studio import load_app
    app = load_app()
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


@pytest.fixture(scope="module")
def page(server):
    with pw.sync_playwright() as p:
        browser = None
        for exe in [c for c in CHROMIUM if c and Path(c).exists()] + [None]:
            try:
                browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
                break
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
        if browser is None:
            pytest.skip("no Chromium available for Playwright")
        pg = browser.new_page(viewport={"width": 1500, "height": 920})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_selector("[data-testid=palette]")
        pg.errors = errors
        yield pg
        browser.close()


def shot(page, name, **kw):
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / name), **kw)


FLOW = {"schema": "pinneapple-flow/1", "id": "gui-heat", "name": "GUI heat", "description": "smoke test",
        "nodes": [{"id": "geo", "type": "geometry.domain", "params": {"dimension": 1, "t_range": [0, 0.3]},
                   "position": {"x": 0, "y": 0}, "label": "Domain"},
                  {"id": "eq", "type": "equations.system", "params": {"template": "heat"},
                   "position": {"x": 280, "y": 0}, "label": "Heat equations"},
                  {"id": "fd", "type": "solver.classical", "params": {"method": "fd", "n": 41, "n_t": 11},
                   "position": {"x": 560, "y": 0}, "label": "FD"},
                  {"id": "val", "type": "validate.compare", "params": {"reference": "exact"},
                   "position": {"x": 840, "y": 0}, "label": "Validation"}],
        "edges": [{"source": "geo", "source_port": "geometry", "target": "eq", "target_port": "geometry"},
                  {"source": "eq", "source_port": "problem", "target": "fd", "target_port": "problem"},
                  {"source": "fd", "source_port": "solution", "target": "val", "target_port": "candidates"},
                  {"source": "eq", "source_port": "problem", "target": "val", "target_port": "problem"}],
        "meta": {}}


def test_drag_from_palette_creates_a_node(page):
    before = page.locator(".react-flow__node").count()
    page.drag_and_drop("[data-testid='palette-geometry.domain']", "[data-testid=canvas]",
                       target_position={"x": 400, "y": 300})
    page.wait_for_function(f"document.querySelectorAll('.react-flow__node').length > {before}")
    assert page.locator("[data-testid=param-panel]").is_visible()


def test_import_run_and_results(page, tmp_path):
    f = tmp_path / "gui.json"
    f.write_text(json.dumps(FLOW))
    page.set_input_files("input[type=file]", str(f))
    page.wait_for_selector("[data-testid='node-FD']")
    page.wait_for_selector("[data-testid=validity]:has-text('valid')")
    page.click("[data-testid=run]")
    page.wait_for_function("""() => [...document.querySelectorAll('[data-testid=node-status]')]
                              .every(e => ['done','cached'].includes(e.textContent))""", timeout=120_000)
    page.click("[data-testid='node-Validation']")
    page.click("[data-testid=tab-results]")
    page.wait_for_selector("[data-testid=metrics]")
    assert "FD_rel_l2" in page.inner_text("[data-testid=metrics]")
    page.wait_for_selector("[data-testid=figure]")
    shot(page, "flow_studio_results.png")
    assert not page.errors


def test_planner_proposals_open_on_the_canvas(page):
    page.click("[data-testid=open-planner]")
    page.fill("[data-testid=planner-text]", "Transient heat conduction in a 2D plate with its edges held at 0 "
                                            "degrees; compare a PINN with finite differences and show it in 3D")
    page.click("[data-testid=planner-go]")
    page.wait_for_selector("[data-testid=proposal]")
    assert "heuristic" in page.inner_text("[data-testid=planner-label]")
    shot(page, "flow_studio_planner.png")
    page.click("[data-testid=proposal-open] >> nth=0")
    page.wait_for_selector("[data-testid=planner-badge]")
    assert page.locator(".react-flow__node").count() >= 8
    page.wait_for_selector("[data-testid=validity]:has-text('valid')")
    shot(page, "flow_studio_canvas.png")


def test_equation_editor_and_architecture_preview(page):
    page.click("[data-testid=open-lists]")
    page.click("[data-testid=example-planned_hybrid_surrogate_hpo]")
    page.wait_for_selector(".react-flow__node")
    page.click(".react-flow__node:has-text('equations')")
    page.wait_for_selector("[data-testid=eq-preview]")
    assert "wave" in page.inner_text("[data-testid=eq-preview]")
    page.click(".react-flow__node:has-text('HPO')")
    page.wait_for_selector("[data-testid=layer-graph]")
    n0 = page.locator("[data-testid=layer-graph] rect").count()
    page.click("[data-testid=add-block]")
    page.wait_for_function(f"document.querySelectorAll('[data-testid=layer-graph] rect').length > {n0}")
    assert "trainable parameters" in page.inner_text("[data-testid=layer-graph]")
    shot(page, "flow_studio_architecture.png")
