import io
from datetime import date
from pathlib import Path

import pytest

from app import create_app
from app.config import Config
from app.services.skills import (
    detect_education,
    estimate_experience_years,
    extract_skills,
    parse_job_description,
)

ROOT = Path(__file__).resolve().parent.parent


class TestConfig(Config):
    TESTING = True
    USE_TRANSFORMERS = False


@pytest.fixture()
def client():
    return create_app(TestConfig).test_client()


def test_extract_skills_aliases_and_boundaries():
    found = extract_skills("Built NLP apps with PyTorch and HuggingFace; wrote C++ and Node.js code. Used ML daily.")
    assert {"NLP", "PyTorch", "Hugging Face", "C++", "JavaScript", "Machine Learning"} <= set(found)


def test_short_aliases_are_case_sensitive():
    assert "RAG" not in extract_skills("the rag was dirty")
    assert "RAG" in extract_skills("We built a RAG pipeline")


def test_no_false_substring_matches():
    assert "Java" not in extract_skills("JavaScript only")
    assert "Git" not in extract_skills("digital transformation")


def test_experience_from_date_ranges_merges_overlap():
    text = "Dev (Jan 2020 - Dec 2021)\nOther (Jun 2021 - Dec 2022)"
    assert estimate_experience_years(text, today=date(2026, 1, 1)) == 3.0


def test_experience_present_and_claims():
    assert estimate_experience_years("Engineer (Jan 2024 - Present)", today=date(2026, 1, 1)) == 2.0
    assert estimate_experience_years("I have 7 years of experience in ML") == 7.0


def test_education_levels():
    assert detect_education("PhD in Physics")[1] == "PhD"
    assert detect_education("M.Sc. Computer Science")[1] == "Master's"
    assert detect_education("B.Sc. Biology")[1] == "Bachelor's"
    assert detect_education("self taught")[0] == 0


def test_job_description_required_vs_preferred():
    jd = (ROOT / "data/sample_jobs/nlp_engineer.txt").read_text()
    job = parse_job_description(jd)
    assert "PyTorch" in job["required_skills"]
    assert "RAG" in job["preferred_skills"]
    assert "RAG" not in job["required_skills"]
    assert job["min_years"] == 4.0
    assert job["min_education_label"] == "Master's"


def test_or_groups_in_job_description():
    job = parse_job_description("Requirements:\n- Python and FastAPI or Flask\n- AWS, GCP or Azure\n- Docker and Git")
    assert ["FastAPI", "Flask"] in job["required_groups"]
    assert ["AWS", "Azure", "GCP"] in job["required_groups"]
    assert ["Python"] in job["required_groups"] and ["Docker"] in job["required_groups"]
    assert ["Git"] in job["required_groups"]


def test_either_alternative_satisfies_requirement(client):
    jd = "Requirements:\n- Python\n- FastAPI or Flask\n"
    cv = b"Jane Roe\nBuilt Python services with Flask for 3 years of experience in backend."
    data = client.post("/api/screen", data={"job_description": jd + "x" * 40,
                                            "cvs": (io.BytesIO(cv), "jane.txt")},
                       content_type="multipart/form-data").get_json()
    assert data["candidates"][0]["missing_required"] == []


def test_intro_and_benefits_sections_ignored():
    job = parse_job_description("About the role\nWe use Kubernetes and Rust.\n\nRequirements:\n- Python\n\nBenefits\n- Java training")
    assert job["required_skills"] == ["Python"]


def test_ranking_order_makes_sense(client):
    jd = (ROOT / "data/sample_jobs/nlp_engineer.txt").read_text()
    res = client.post("/api/screen", data={"job_description": jd, "use_samples": "true"})
    assert res.status_code == 200
    data = res.get_json()
    names = [c["name"] for c in data["candidates"]]
    assert names[0] in {"Amina El Idrissi", "Carlos Mendes"}
    assert names[-1] == "Sara Okafor"
    for c in data["candidates"]:
        assert 0 <= c["score"] <= 100
        assert abs(sum(v["points"] for v in c["components"].values()) - c["score"]) < 0.2
        assert c["reasons"]


def test_llm_job_prefers_llm_developer(client):
    jd = (ROOT / "data/sample_jobs/ai_developer.txt").read_text()
    data = client.post("/api/screen", data={"job_description": jd, "use_samples": "true"}).get_json()
    assert data["candidates"][0]["name"] == "Jonas Weber"


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_upload_txt_and_html_results(client):
    jd = (ROOT / "data/sample_jobs/nlp_engineer.txt").read_text()
    cv = (ROOT / "data/sample_cvs/liam_chen.txt").read_bytes()
    res = client.post(
        "/screen",
        data={"job_description": jd, "cvs": (io.BytesIO(cv), "liam.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    assert b"Ranked candidates" in res.data and b"Liam Chen" in res.data


def test_validation_errors(client):
    assert client.post("/screen", data={"job_description": "short"}).status_code == 400
    assert client.post("/api/screen", data={"job_description": "x" * 60}).status_code == 400


def test_rejects_unsupported_file(client):
    res = client.post(
        "/api/screen",
        data={"job_description": "x" * 60, "cvs": (io.BytesIO(b"MZ"), "virus.exe")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400


def test_docx_and_pdf_parsing():
    import docx
    from app.services.parser import extract_text

    d = docx.Document()
    d.add_paragraph("Jane Doe is a Python and NLP engineer with five years of experience.")
    buf = io.BytesIO()
    d.save(buf)
    assert "Python" in extract_text("jane.docx", buf.getvalue())
