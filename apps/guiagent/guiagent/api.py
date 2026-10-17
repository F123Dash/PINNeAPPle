"""Engineering GUI Agent: web API + single-page UI. Give it a task (a sentence or a structured workflow); it runs the
work in FreeCAD, CfdOF, OpenFOAM and ParaView inside this container and hands back the files plus the evidence of
every step: screenshots, logs, commands, produced files with SHA-256, checks and the acceptance criteria.

All the work is done by ``pinneapple_tools.automation``; this module only exposes it over HTTP.

Run:  uvicorn guiagent.api:app --port 8115   (from apps/guiagent)

Environment (all optional):
  GUIAGENT_USER / GUIAGENT_PASSWORD   HTTP Basic login on everything except /health
  GUIAGENT_MODE                       public (default: only the committed workflows) or selfhosted (also uploads)
  GUIAGENT_DATA                       folder for runs and uploaded workflows (default ./data)
  GUIAGENT_MAX_JOBS                   concurrent runs (default 1); more requests get 429
  GUIAGENT_KEEP_RUNS                  runs kept on disk, oldest deleted first (default 40)
  GUIAGENT_OLLAMA_URL / GUIAGENT_OLLAMA_MODEL   optional local LLM for reading sentences (rule-based without it)
"""
from __future__ import annotations

import io
import os
import re
import secrets
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pinneapple_tools.automation import (
    Runner,
    RunRecord,
    Toolchain,
    Workflow,
    WorkflowError,
    catalog,
    interpret,
    known_actions,
    plan,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "_shared"))
from appkit import install  # noqa: E402

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
MODE = (os.environ.get("GUIAGENT_MODE") or "public").lower()
DATA = Path(os.environ.get("GUIAGENT_DATA") or HERE.parent / "data").resolve()
RUNS = DATA / "runs"
UPLOADED = DATA / "workflows"
MAX_JOBS = int(os.environ.get("GUIAGENT_MAX_JOBS") or 1)
KEEP_RUNS = int(os.environ.get("GUIAGENT_KEEP_RUNS") or 40)
RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

app = FastAPI(title="Engineering GUI Agent", version=VERSION,
              description="Run a task in engineering software (FreeCAD, CfdOF, OpenFOAM, ParaView) and get the files "
                          "plus the evidence of every step. POST /api/runs starts a run of a committed workflow; it "
                          "stops before expensive steps until POST /api/runs/{id}/approve. GET /api/runs/{id} follows "
                          "it; /report, /study_result and /bundle deliver the evidence.")
app.add_middleware(GZipMiddleware, minimum_size=2000)
install(app, prefix="GUIAGENT")

_jobs = threading.BoundedSemaphore(MAX_JOBS)
_lock = threading.Lock()
_active: dict[str, str] = {}                    # run id -> step being run (for the UI while a thread works)
_toolchain = Toolchain.discover()


def _llm():
    model = os.environ.get("GUIAGENT_OLLAMA_MODEL")
    if not model:
        return None
    from pinneapple_data.formfill.llm import OllamaClient
    return OllamaClient(os.environ.get("GUIAGENT_OLLAMA_URL") or "http://ollama:11434", model, timeout=120)


def _catalog() -> dict[str, Workflow]:
    cat = catalog()
    if MODE == "selfhosted" and UPLOADED.is_dir():
        cat.update(catalog(UPLOADED))
    return cat


def _workflow(wid: str) -> Workflow:
    cat = _catalog()
    if wid not in cat:
        raise HTTPException(404, f"No workflow {wid!r}. Available: {', '.join(sorted(cat))}.")
    return cat[wid]


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id) or not (RUNS / run_id / "run.json").is_file():
        raise HTTPException(404, "No such run.")
    return RUNS / run_id


def _prune() -> None:
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and RUN_ID.match(p.name)), key=lambda p: p.name)
    for p in runs[:-KEEP_RUNS] if len(runs) > KEEP_RUNS else []:
        if p.name not in _active:
            shutil.rmtree(p, ignore_errors=True)


def _slot() -> None:
    """Take one of the MAX_JOBS run slots, or answer 429."""
    if not _jobs.acquire(blocking=False):
        raise HTTPException(429, f"The agent is busy with another run ({MAX_JOBS} at a time on this server). "
                                 "Please retry in a minute.")


def _launch(fn, run_id: str) -> None:
    """Run ``fn`` in a background thread holding the slot taken by :func:`_slot`."""
    def work():
        try:
            fn()
        finally:
            with _lock:
                _active.pop(run_id, None)
            _jobs.release()

    with _lock:
        _active[run_id] = "starting"
    threading.Thread(target=work, daemon=True).start()


def _on_event(e: dict) -> None:
    if e.get("step"):
        with _lock:
            _active[e["run_id"]] = e["step"]


# ----------------------------------------------------------------------------------------------------- schemas
class InterpretIn(BaseModel):
    task: str = Field(..., max_length=4000, description="A sentence, or a structured request as JSON/YAML text")


class RunIn(BaseModel):
    workflow: str = Field("channel_obstacle", description="Id of a committed workflow (GET /api/workflows)")
    inputs: dict = Field(default_factory=dict, description="Input values; the workflow's bounds apply")
    task: str | None = Field(None, max_length=4000, description="Optional sentence, read instead of workflow+inputs")


class Decision(BaseModel):
    step: str
    by: str = Field("", max_length=80, description="Who decides (kept in the evidence)")
    reason: str = Field("", max_length=400)


# ------------------------------------------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/api/meta")
def meta():
    missing = _toolchain.missing(["freecad", "cfdof", "openfoam", "gmsh", "paraview", "xvfb", "xdotool"])
    return {"version": VERSION, "mode": MODE, "max_jobs": MAX_JOBS, "busy": len(_active),
            "llm": bool(os.environ.get("GUIAGENT_OLLAMA_MODEL")), "tools": _toolchain.versions(),
            "missing_tools": missing, "uploads": MODE == "selfhosted"}


@app.get("/api/workflows")
def workflows():
    """Every workflow this server runs, with its inputs, steps (and whether their application is installed),
    acceptance criteria and limits."""
    out = []
    for wf in _catalog().values():
        d = wf.to_dict()
        d["plan"] = plan(wf, _toolchain)
        out.append(d)
    return out


@app.post("/api/interpret")
def api_interpret(body: InterpretIn):
    """Read a task: which committed workflow it asks for and with which inputs. Nothing is run."""
    try:
        it = interpret(body.task, _catalog(), llm=_llm())
    except (WorkflowError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    if it.workflow.id not in _catalog():
        raise HTTPException(403, "Only committed workflows run on this server; a structured workflow is read but not "
                                 "run here (self-hosted mode can add workflows).")
    return {**it.to_dict(), "resolved_inputs": it.workflow.resolve_inputs(it.inputs)}


@app.post("/api/runs")
def start_run(body: RunIn):
    """Start a run. It goes on in the background until it finishes, fails, or reaches a step that needs a person's
    approval (the solver step of the example); follow it with GET /api/runs/{id}."""
    try:
        if body.task:
            it = interpret(body.task, _catalog(), llm=_llm())
            wf, inputs = it.workflow, it.inputs
            if wf.id not in _catalog():
                raise HTTPException(403, "Only committed workflows run on this server.")
        else:
            wf, inputs = _workflow(body.workflow), body.inputs
        runner_inputs = wf.resolve_inputs(inputs)
    except WorkflowError as e:
        raise HTTPException(422, str(e)) from e
    RUNS.mkdir(parents=True, exist_ok=True)
    _slot()
    try:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
        runner = Runner(wf, RUNS / run_id, inputs=runner_inputs, toolchain=_toolchain, on_event=_on_event)
        _prune()
    except BaseException:
        _jobs.release()
        raise
    _launch(runner.run, run_id)
    return {"run_id": run_id, "workflow": wf.id, "inputs": runner_inputs}


@app.get("/api/runs")
def list_runs(limit: int = 20):
    if not RUNS.is_dir():
        return []
    out = []
    for p in sorted((p for p in RUNS.iterdir() if RUN_ID.match(p.name)), reverse=True)[:max(1, min(limit, 100))]:
        try:
            r = RunRecord.load(p)
        except (OSError, ValueError):
            continue
        out.append({"run_id": r.run_id, "workflow": r.workflow.get("id"), "title": r.workflow.get("title"),
                    "inputs": r.inputs, "state": r.state, "message": r.message, "created": r.created,
                    "finished": r.finished, "active": r.run_id in _active})
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    """The live record: state, every step (state, duration, commands, logs, artifacts, screenshots, checks,
    outputs, confirmation), the acceptance criteria and the tools used."""
    d = RunRecord.load(_run_dir(run_id)).to_dict()
    d["active"] = run_id in _active
    d["current_step"] = _active.get(run_id)
    d.pop("folder", None)
    return d


def _resume(run_id: str, approve: list[str], by: str) -> None:
    folder = _run_dir(run_id)
    rec = RunRecord.load(folder)
    wf = _workflow(rec.workflow["id"])
    _slot()
    try:
        runner = Runner(wf, folder, toolchain=_toolchain, on_event=_on_event)
    except BaseException:
        _jobs.release()
        raise
    _launch(lambda: runner.run(approve=approve, approver=by), run_id)


@app.post("/api/runs/{run_id}/approve")
def approve(run_id: str, body: Decision):
    """Approve the step the run is waiting on; the run continues in the background."""
    rec = RunRecord.load(_run_dir(run_id))
    if run_id in _active:
        raise HTTPException(409, "This run is already running.")
    step = next((s for s in rec.steps if s.id == body.step), None)
    if step is None or step.state != "awaiting_confirmation":
        raise HTTPException(409, f"Step {body.step!r} is not waiting for a confirmation.")
    _resume(run_id, [body.step], body.by.strip() or "web user")
    return {"run_id": run_id, "approved": body.step}


@app.post("/api/runs/{run_id}/decline")
def decline(run_id: str, body: Decision):
    """Refuse the confirmation: the run ends there, and the evidence says who declined and why."""
    folder = _run_dir(run_id)
    if run_id in _active:
        raise HTTPException(409, "This run is already running.")
    rec = RunRecord.load(folder)
    try:
        rec = Runner(_workflow(rec.workflow["id"]), folder, toolchain=_toolchain).decline(
            body.step, by=body.by.strip() or "web user", reason=body.reason)
    except WorkflowError as e:
        raise HTTPException(409, str(e)) from e
    return {"run_id": run_id, "state": rec.state}


_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
          ".csv": "text/csv", ".html": "text/html", ".txt": "text/plain", ".log": "text/plain"}


@app.get("/api/runs/{run_id}/files/{path:path}")
def run_file(run_id: str, path: str):
    """A file of the run (screenshots, logs, exported fields, the FreeCAD documents)."""
    root = _run_dir(run_id).resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise HTTPException(404, "No such file in this run.")
    media = _MEDIA.get(p.suffix.lower()) or ("text/plain" if p.name.startswith("log.") else "application/octet-stream")
    return FileResponse(p, media_type=media)


@app.get("/api/runs/{run_id}/report")
def report(run_id: str):
    """The evidence report: one self-contained HTML file (print it to PDF from the browser)."""
    p = _run_dir(run_id) / "report.html"
    if not p.is_file():
        raise HTTPException(404, "The report is written when the run stops.")
    return FileResponse(p, media_type="text/html")


@app.get("/api/runs/{run_id}/study_result")
def study_result(run_id: str):
    """The run as a StudyResult (schema pinneapple.study_result/0), for the Evidence Report app."""
    p = _run_dir(run_id) / "study_result.json"
    if not p.is_file():
        raise HTTPException(404, "The StudyResult is written when the run stops.")
    return FileResponse(p, media_type="application/json")


@app.get("/api/runs/{run_id}/bundle")
def bundle(run_id: str):
    """Everything the run produced as a zip: record, report, StudyResult, FreeCAD documents, the OpenFOAM case at
    its last time, logs, screenshots, exported fields."""
    root = _run_dir(run_id)
    if run_id in _active:
        raise HTTPException(409, "The run is still going; download it when it stops.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                z.write(p, f"{run_id}/{p.relative_to(root).as_posix()}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="guiagent-{run_id}.zip"'})


@app.post("/api/workflows")
async def upload_workflow(file: UploadFile = File(...)):  # noqa: B008  (FastAPI idiom)
    """Self-hosted mode only: add a workflow (a zip with workflow.yaml and the files it uses, e.g. your FreeCAD
    project and macro). A public server refuses: macros are code."""
    if MODE != "selfhosted":
        raise HTTPException(403, "This is the public demo: only the committed workflows run here. Uploading "
                                 "workflows and macros is available in self-hosted mode (GUIAGENT_MODE=selfhosted).")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Workflow archives are limited to 50 MB.")
    tmp = UPLOADED / (".upload-" + secrets.token_hex(4))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                target = (tmp / name).resolve()
                if not target.is_relative_to(tmp.resolve()):
                    raise HTTPException(422, f"Unsafe path in the archive: {name}")
            z.extractall(tmp)
        found = sorted(tmp.rglob("workflow.y*ml")) + sorted(tmp.rglob("workflow.json"))
        if not found:
            raise HTTPException(422, "The archive has no workflow.yaml.")
        wf = Workflow.load(found[0])
        errors = wf.validate(known_actions())
        if errors:
            raise HTTPException(422, "; ".join(errors))
        if wf.id in catalog():
            raise HTTPException(409, f"{wf.id!r} is a committed workflow; choose another id.")
        dest = UPLOADED / wf.id
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(found[0].parent), dest)
    except zipfile.BadZipFile as e:
        raise HTTPException(422, "Upload a zip archive.") from e
    except WorkflowError as e:
        raise HTTPException(422, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"workflow": wf.id, "title": wf.title}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
