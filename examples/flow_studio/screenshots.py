"""Screenshots of Flow Studio for the docs (docs/flow_studio/): the canvas of a planned flow, a node's result panel
with the three.js viewer, the PINN node's architecture and validation, and the equation editor.

Start the studio first (``python -m pinneapple_flow studio --no-browser``), then::

    python examples/flow_studio/screenshots.py [--url http://127.0.0.1:8765] [--chromium /path/to/chrome]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / "docs" / "flow_studio"


def wait_done(page, timeout_ms=900_000):
    page.wait_for_function("""() => [...document.querySelectorAll('[data-testid=node-status]')]
                              .every(e => ['done','cached','failed'].includes(e.textContent))""", timeout=timeout_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chromium if Path(a.chromium).exists() else None)
        page = b.new_page(viewport={"width": 1600, "height": 960}, device_scale_factor=1)
        page.goto(a.url)
        page.wait_for_selector("[data-testid=palette]")
        # 1. the planner and the canvas it fills
        page.click("[data-testid=open-planner]")
        page.fill("[data-testid=planner-text]", "Steady heat conduction in a square plate with internal heat "
                                                "generation and the edges held at 0 degrees; compare finite elements,"
                                                " finite differences and a PINN. Then switch the plate from 2D to 3D.")
        page.click("[data-testid=planner-go]")
        page.wait_for_selector("[data-testid=proposal]")
        page.screenshot(path=str(OUT / "flow_studio_planner.png"))
        page.click("[data-testid=proposal-open] >> nth=0")
        page.wait_for_selector("[data-testid=planner-badge]")
        page.click(".react-flow__node:has-text('equations')")
        page.wait_for_selector("[data-testid=eq-preview]")
        page.screenshot(path=str(OUT / "flow_studio_canvas.png"))
        # 2. run it and open the results
        page.click("[data-testid=run]")
        wait_done(page)
        page.click(".react-flow__node:has-text('3-D viewer')")
        page.click("[data-testid=tab-results]")
        page.wait_for_selector("[data-testid=viewer3d] canvas", timeout=60_000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "flow_studio_result_3d.png"))
        page.click(".react-flow__node:has-text('Validation')")
        page.wait_for_selector("[data-testid=figure]")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "flow_studio_result_validation.png"))
        page.click(".react-flow__node:has-text('PINN')")
        page.wait_for_selector("[data-testid=layer-graph]")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "flow_studio_result_pinn.png"))
        # 3. the architecture editor of a surrogate
        page.click("[data-testid=open-lists]")
        page.click("[data-testid=example-planned_hybrid_surrogate_hpo]")
        page.wait_for_selector(".react-flow__node")
        page.click(".react-flow__node:has-text('HPO')")
        page.click("[data-testid=tab-params]")
        page.wait_for_selector("[data-testid=layer-graph]")
        page.click("[data-testid=add-block]")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "flow_studio_architecture.png"))
        b.close()
    print(f"screenshots in {OUT}")


if __name__ == "__main__":
    main()
