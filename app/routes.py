from __future__ import annotations

from pathlib import Path

from flask import (Blueprint, current_app, jsonify, render_template, request)
from werkzeug.utils import secure_filename

from .services.parser import CVParseError, extract_text

bp = Blueprint("main", __name__)


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def _sample_jobs() -> dict[str, str]:
    jobs = {}
    for path in sorted(Path(current_app.config["SAMPLE_JOB_DIR"]).glob("*.txt")):
        jobs[path.stem.replace("_", " ").title()] = path.read_text(encoding="utf-8")
    return jobs


def _load_sample_cvs() -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8"))
            for p in sorted(Path(current_app.config["SAMPLE_CV_DIR"]).glob("*.txt"))]


def _collect_cvs(files, use_samples: bool) -> tuple[list[tuple[str, str]], list[str]]:
    cvs: list[tuple[str, str]] = []
    errors: list[str] = []
    seen: set[str] = set()

    uploads = [f for f in files if f and f.filename]
    if len(uploads) > current_app.config["MAX_CVS_PER_REQUEST"]:
        errors.append(f"Too many files; max {current_app.config['MAX_CVS_PER_REQUEST']} per request. Extra files ignored.")
        uploads = uploads[: current_app.config["MAX_CVS_PER_REQUEST"]]

    for f in uploads:
        name = secure_filename(f.filename) or "cv"
        if not _allowed(name):
            errors.append(f"{f.filename}: unsupported type (allowed: PDF, DOCX, TXT, MD).")
            continue
        if name in seen:
            stem, _, ext = name.rpartition(".")
            name = f"{stem}_{len(seen)}.{ext}"
        seen.add(name)
        try:
            cvs.append((name, extract_text(name, f.read())))
        except CVParseError as exc:
            errors.append(str(exc))

    if use_samples:
        cvs.extend(_load_sample_cvs())
    return cvs, errors


@bp.get("/")
def index():
    return render_template("index.html", jobs=_sample_jobs())


@bp.post("/screen")
def screen():
    job_text = (request.form.get("job_description") or "").strip()
    use_samples = request.form.get("use_samples") == "on"
    cvs, errors = _collect_cvs(request.files.getlist("cvs"), use_samples)

    if len(job_text) < 40:
        errors.append("Please paste a job description (at least a couple of sentences).")
    if not cvs:
        errors.append("No readable CVs. Upload files or tick 'use sample CVs'.")
    if len(job_text) < 40 or not cvs:
        return render_template("index.html", jobs=_sample_jobs(), errors=errors, job_text=job_text), 400

    result = current_app.extensions["ranker"].rank(job_text, cvs)
    return render_template("results.html", result=result, errors=errors)


@bp.post("/api/screen")
def api_screen():
    job_text = (request.form.get("job_description") or "").strip()
    if len(job_text) < 40:
        return jsonify(error="job_description is required (min 40 chars)"), 400
    cvs, errors = _collect_cvs(request.files.getlist("cvs"), request.form.get("use_samples") == "true")
    if not cvs:
        return jsonify(error="no readable CVs", details=errors), 400
    result = current_app.extensions["ranker"].rank(job_text, cvs)
    result["warnings"] = errors
    return jsonify(result)


@bp.get("/api/health")
def health():
    scorer = current_app.extensions["ranker"].scorer
    return jsonify(status="ok", backend=scorer.backend, note=scorer.note)
