"""Electromagnetics verification table: every solver and reference implementation of
``pinneapple_simulation.electromagnetics`` against an independent result, with measured errors and orders.

The table in docs/guides/electromagnetics.md is the output of this script.

    python examples/electrodynamics/07_em_verification.py            # all cases (about 30 s)
    python examples/electrodynamics/07_em_verification.py --quick    # coarser refinement ladders
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pinneapple_simulation.electromagnetics import verification

OUT = Path(__file__).parent / "results"

if __name__ == "__main__":
    results = verification.run(quick="--quick" in sys.argv)
    print(verification.to_markdown(results))
    for r in results:
        if r.detail:
            print(f"- {r.name}: {r.detail}")
    OUT.mkdir(exist_ok=True)
    (OUT / "em_verification.json").write_text(json.dumps(
        [{"case": r.name, "quantity": r.what, "reference": r.reference, "error": r.error, "orders": r.orders,
          "expected_order": r.expected_order, "passed": r.passed, "detail": r.detail, "seconds": r.seconds}
         for r in results], indent=1))
    sys.exit(0 if all(r.passed for r in results) else 1)
