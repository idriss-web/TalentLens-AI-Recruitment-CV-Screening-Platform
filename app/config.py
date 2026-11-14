import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
    MAX_CVS_PER_REQUEST = 50

    USE_TRANSFORMERS = os.environ.get("USE_TRANSFORMERS", "1") == "1"
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    WEIGHTS = {
        "required_skills": 0.40,
        "semantic": 0.25,
        "preferred_skills": 0.15,
        "experience": 0.15,
        "education": 0.05,
    }

    SAMPLE_CV_DIR = BASE_DIR / "data" / "sample_cvs"
    SAMPLE_JOB_DIR = BASE_DIR / "data" / "sample_jobs"
