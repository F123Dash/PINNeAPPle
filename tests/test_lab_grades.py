"""Lab tiers A-F: one fixture per rule, deterministic re-curation (of the committed lab too), every seal rendered,
no way to set a tier by hand, and the tier filter for stamping."""
import inspect
import json
import os

import numpy as np
import pytest

from pinneapple_lab import Experiment, LabStore, register, sweep
from pinneapple_lab.curation import TIER_HISTORY, TIERS, curate, problem
from pinneapple_lab.seal import (
    SEALABLE,
    TIER_COLOURS,
    TIER_INK,
    TIER_LINE,
    SealError,
    contrast_ratio,
    legend_seals,
    render_legend,
    render_seal,
    seal_for,
    stamp_all,
    stamp_file,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@register
class _Grade(Experiment):
    name = "_test_grade"
    description = ("One case per curation rule, A to F: what each case records decides its letter, nothing else "
                   "(fixtures of tests/test_lab_grades.py).")
    references = ["Fixture: no external reference"]
    params = {"case": "D"}
    case_param = "case"

    def run(self, ctx):
        c = ctx.params["case"]
        if c == "A":                                      # reference + physics, three figures, documented
            ctx.check("deflection_vs_beam_theory", value=1.001, reference=1.0, rtol=0.01, kind="reference")
            ctx.check("energy_balance", True, kind="physics")
            for k in range(3):
                ctx.image(f"field_{k}", np.full((8, 8, 3), 40 * k, np.uint8))
        elif c == "B":
            ctx.check("energy_balance", True, kind="physics")
        elif c == "C":                                    # only a second code that shares the inputs
            ctx.check("torque_vs_second_code", True, kind="verification")
        elif c == "D":                                    # runs, consistent, nothing that answers a stage
            ctx.check("output_written", True, kind="sanity")
        elif c == "E_reference":                          # large error vs the reference, physics still holds
            ctx.check("drag_vs_wind_tunnel", False, kind="reference", detail="Ahmed 1984")
            ctx.check("energy_balance", True, kind="physics")
        elif c == "E_generalization":                     # outside the validity range
            ctx.check("error_on_unseen_designs", False, kind="generalization")
            ctx.check("energy_balance", True, kind="physics")
        elif c == "E_baseline":
            ctx.check("beats_persistence", False, kind="baseline")
            ctx.check("drag_vs_wind_tunnel", True, kind="reference")
        elif c == "F_crashed":
            raise RuntimeError("solver blew up at step 12")
        elif c == "F_diverged":
            ctx.check("solution_finite", False, kind="sanity")
            ctx.check("drag_vs_wind_tunnel", True, kind="reference")
        elif c == "F_physics":                            # failed conservation, even with a matching reference
            ctx.check("mass_balance", False, kind="physics")
            ctx.check("drag_vs_wind_tunnel", True, kind="reference")
        elif c == "F_contradicts":                        # every reference check fails, nothing else independent
            ctx.check("drag_vs_wind_tunnel", False, kind="reference")
            ctx.check("lift_vs_wind_tunnel", False, kind="reference")
            ctx.check("loss_decreased", True, kind="sanity")


CASES = {"A": "A", "B": "B", "C": "C", "D": "D", "E_reference": "E", "E_generalization": "E", "E_baseline": "E",
         "F_crashed": "F", "F_diverged": "F", "F_physics": "F", "F_contradicts": "F"}


@pytest.fixture(scope="module")
def graded(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("grades"))
    sweep("_test_grade", grid={"case": list(CASES)}, root=root)
    return root, curate(LabStore(root))


# ---------------------------------------------------------------------------------------------- the rules
@pytest.mark.parametrize("case", list(CASES))
def test_each_rule_gives_its_letter(graded, case):
    _, out = graded
    item = out["items"][f"_test_grade/{case}"]
    assert item["tier"] == CASES[case] and item["tier_name"] == TIERS[CASES[case]]
    hl = item["headline"]
    if CASES[case] in ("E", "F"):                       # the main problem and the next step, from the curation
        assert hl["kind"] == "failing" and hl["rule"] == case.split("_", 1)[1]
        assert hl["hint"].startswith({"E": "D: ", "F": "E: "}[CASES[case]]) and hl["hint_pt"][:3] == hl["hint"][:3]
        assert item["stages"]["decision"]["status"] == "failed"
    else:
        assert hl.get("kind") != "failing"


def test_problem_lines_name_the_failing_check(graded):
    _, out = graded
    it = out["items"]
    assert it["_test_grade/F_crashed"]["headline"]["text"] == "crashed · RuntimeError: solver blew up at step 12"
    assert it["_test_grade/F_physics"]["headline"]["check"] == "mass_balance"
    assert it["_test_grade/F_contradicts"]["headline"]["text"].endswith("(+1 more)")
    assert it["_test_grade/E_reference"]["headline"]["text"] == "drag vs wind tunnel · Ahmed 1984"
    assert it["_test_grade/C"]["headline"]["kind"] == "verification"
    assert it["_test_grade/D"]["headline"]["text"] == ""
    assert problem("completed", [{"name": "x", "passed": True, "kind": "physics"}]) is None


def test_old_tiers_map_onto_the_new_letters():
    assert TIER_HISTORY == {"A": ("A",), "B": ("B",), "C": ("C", "D"), "D": ("E", "F")}
    assert list(TIERS) == list(SEALABLE) == ["A", "B", "C", "D", "E", "F"]


def test_experiment_counts_every_letter_and_one_good_run_is_not_a_result(graded):
    _, out = graded
    e = out["experiments"]["_test_grade"]
    assert e["tier_counts"] == {"A": 1, "B": 1, "C": 1, "D": 1, "E": 3, "F": 4}
    assert e["tier"] == "B"                    # best run A, but fewer than half of its 11 runs are usable


# ---------------------------------------------------------------------------------------------- determinism
def test_recuration_reproduces_the_letters(graded):
    root, first = graded
    again = curate(LabStore(root))
    strip = lambda d: {k: v for k, v in d.items() if k != "generated"}  # noqa: E731
    assert json.dumps(strip(again), sort_keys=True, default=str) == json.dumps(strip(first), sort_keys=True, default=str)


def test_committed_lab_curation_is_reproduced(tmp_path):
    """Curating the committed lab runs again gives the committed letters and headlines (lab/curation.json)."""
    runs = os.path.join(REPO, "lab", "runs")
    if not os.path.isdir(runs):
        pytest.skip("no lab runs in this checkout")
    os.symlink(runs, tmp_path / "runs")
    store = LabStore(str(tmp_path))
    store.reindex()
    fresh = curate(store)
    committed = json.load(open(os.path.join(REPO, "lab", "curation.json")))
    assert set(fresh["items"]) == set(committed["items"])
    for key, it in committed["items"].items():
        assert (fresh["items"][key]["tier"], fresh["items"][key]["headline"]) == (it["tier"], it["headline"]), key


# ---------------------------------------------------------------------------------------------- no tier by hand
def test_the_tier_cannot_be_set_by_hand(graded, tmp_path, capsys):
    root, _ = graded
    for fn in (seal_for, stamp_all, stamp_file, render_seal):
        assert "tier" not in inspect.signature(fn).parameters
    with pytest.raises(TypeError):
        seal_for("_test_grade/F_physics", root, tier="A")  # noqa
    assert seal_for("_test_grade/F_physics", root).tier == "F"
    cur = json.load(open(os.path.join(root, "curation.json")))
    cur["items"]["_test_grade/D"]["tier"] = "A+"          # a hand-edited record is refused, not sealed
    with pytest.raises(SealError):
        seal_for("_test_grade/D", cur)
    from pinneapple_lab.__main__ import main as cli
    assert cli(["--root", root, "seal", "_test_grade/D", "--tiers", "A", "-o", str(tmp_path / "s.svg")]) == 2
    assert not os.path.exists(tmp_path / "s.svg")      # a tier option filters; it never changes the letter


# ---------------------------------------------------------------------------------------------- rendering
@pytest.mark.parametrize("tier", list(SEALABLE))
@pytest.mark.parametrize("lang", ["en", "pt"])
def test_every_letter_renders(graded, tier, lang):
    root, out = graded
    key = next(k for k, v in out["items"].items() if v["tier"] == tier)
    s = seal_for(key, root)
    full = render_seal(s, height=96, lang=lang, compact=False)
    small = render_seal(s, height=40, lang=lang)
    svg_full, svg_small = render_seal(s, fmt="svg", lang=lang), render_seal(s, fmt="svg", height=40, lang=lang)
    assert full.mode == "RGBA" and np.asarray(full)[..., 3].max() == 255
    assert f">{tier}</text>" in svg_full and TIER_COLOURS[tier] in svg_full
    if tier in ("E", "F"):                     # the failing check and the next step, always: no compact form
        reason, hint = s.text(lang)
        assert reason and hint
        assert full.height > 96 and small.height > 40
        for svg in (svg_full, svg_small, render_seal(s, fmt="svg", reason=False, compact=True, lang=lang)):
            assert "✗" in svg and "↑" in svg and hint.split(": ", 1)[1][:12] in svg
    else:
        assert full.height == 96 and small.height == 40 and small.width < full.width
        assert "TIER" not in svg_small


def test_colours_are_readable():
    for t in SEALABLE:
        assert contrast_ratio(TIER_INK[t], TIER_COLOURS[t]) >= 4.5           # the letter on its disc
        assert contrast_ratio(TIER_LINE[t]) >= 4.5                            # the tier line on the badge
    assert len(set(TIER_COLOURS.values())) == 6


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_legend(lang):
    assert [s.tier for s in legend_seals(lang)] == list(SEALABLE)
    im = render_legend(lang)
    assert im.width > 600 and im.height > 6 * 72


# ---------------------------------------------------------------------------------------------- stamping and --tiers
def _png(path):
    from PIL import Image
    Image.fromarray(np.full((240, 360, 3), 255, np.uint8)).save(path)


def test_stamp_any_letter_and_the_tiers_filter(graded, tmp_path, capsys):
    root, _ = graded
    from pinneapple_lab.__main__ import main as cli
    src = str(tmp_path / "plot.png")
    _png(src)
    for tier, key in (("A", "_test_grade/A"), ("D", "_test_grade/D"), ("F", "_test_grade/F_crashed")):
        out = stamp_file(src, seal_for(key, root), str(tmp_path / f"stamped_{tier}.png"), lang="pt")
        assert os.path.getsize(out) > 0
    assert cli(["--root", root, "stamp", "_test_grade/E_baseline", src, "-o", str(tmp_path / "e.png")]) == 0
    assert cli(["--root", root, "stamp", "_test_grade/E_baseline", src, "--tiers", "A,B",
                "-o", str(tmp_path / "e_refused.png")]) == 2
    assert not os.path.exists(tmp_path / "e_refused.png")
    assert cli(["--root", root, "stamp", "--all", "--tiers", "A,B", "--out", str(tmp_path / "ab")]) == 0
    man = json.load(open(tmp_path / "ab" / "manifest.json"))
    assert {v["tier"] for v in man["items"].values()} == {"A", "B"}
    assert cli(["--root", root, "stamp", "--all", "--out", str(tmp_path / "all")]) == 0
    man = json.load(open(tmp_path / "all" / "manifest.json"))
    assert {v["tier"] for v in man["items"].values()} == set(SEALABLE)
    assert all(v["hint"] for v in man["items"].values() if v["tier"] in ("E", "F"))
    with pytest.raises(ValueError):
        stamp_all(root, str(tmp_path / "bad"), tiers="A,G")
    assert cli(["legend", "--lang", "pt", "-o", str(tmp_path / "legend.png")]) == 0
