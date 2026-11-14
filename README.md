

https://github.com/user-attachments/assets/1f206c5a-8eb1-4dfd-901d-3ea0606a5391


# TalentLens: AI Recruitment & CV Screening Platform

Upload CVs, compare them against a job description, extract skills and get a **ranked shortlist
with an explanation for every score**. Built with Python, Flask, NLP and Transformers, with
AI Developer and NLP Engineer roles as the default target.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5000, pick **"Load a sample job…"**, tick **"include the built-in sample CVs"** and press **Rank candidates**.

The first start with Transformers enabled downloads `all-MiniLM-L6-v2` (~90 MB).
No GPU needed. To skip torch entirely, set `USE_TRANSFORMERS=0` in the `.env` file;
the app then uses TF-IDF similarity instead.

## How scoring works

Every candidate gets a 0-100 score built from five visible components (weights in `app/config.py`):

| Component | Weight | What it measures |
|---|---|---|
| Required skills | 40% | Share of the JD's required skills found in the CV |
| Semantic fit | 25% | Transformer embedding similarity (chunked JD vs chunked CV) |
| Preferred skills | 15% | Share of "nice to have" skills found |
| Experience | 15% | Years of experience (from date ranges + "N years" claims) vs. the JD minimum |
| Education | 5% | Highest degree detected vs. the JD's requirement |

The results page shows points per component, matched / missing skills, flags (for example
"below the 4+ year requirement") and the top terms shared with the JD, so a recruiter can see *why*
someone ranked where they did.

**Job description parsing**
- Skills under "Nice to have / Preferred / Bonus" count as preferred; sections like "About" and "Benefits" are ignored.
- "FastAPI **or** Flask" is one requirement that either skill satisfies; "Python **and** PyTorch" needs both.

## API

```bash
curl -F "job_description=<data/sample_jobs/nlp_engineer.txt" \
     -F "cvs=@data/sample_cvs/amina_el_idrissi.txt" \
     -F "cvs=@some_candidate.pdf" \
     http://127.0.0.1:5000/api/screen
```

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `POST /screen` | Form submit, returns the HTML results page |
| `POST /api/screen` | JSON ranking (`job_description` + one or more `cvs` files; add `use_samples=true` for the demo CVs) |
| `GET /api/health` | Status and which similarity engine is active |

## Project layout

```
app/
  __init__.py        app factory (loads the embedding model once)
  config.py          weights, limits, model name
  routes.py          web + API routes
  services/
    parser.py        PDF / DOCX / TXT -> text
    skills.py        skill taxonomy, experience, education, JD parsing
    embedder.py      Sentence-Transformers scorer (+ TF-IDF fallback)
    ranker.py        weighted scoring + explanations
  templates/         Jinja pages (no external CDN)
data/                sample jobs and 6 fictional CVs
tests/               pytest suite
run.py               entry point
```

## Customising

- **Skills:** edit `SKILL_TAXONOMY` in `app/services/skills.py` (canonical name -> aliases). Add skills for other roles here.
- **Weights:** edit `WEIGHTS` in `app/config.py` (they are normalised automatically).
- **Model:** set `EMBEDDING_MODEL` to any sentence-transformers model, e.g. `BAAI/bge-small-en-v1.5` or a multilingual one for non-English CVs.
- **Tests:** `pytest`

## Limitations

- Experience is the *total* detected years, not years relevant to the role.
- Scanned/image-only PDFs have no text layer (no OCR included).
- Skill extraction is dictionary-based: it only finds skills in the taxonomy.
- CVs are not stored: they are parsed in memory per request.
- This is a decision-support tool. Rankings can reflect keyword and phrasing bias, so keep a human in the loop for hiring decisions, and check local regulations on automated screening.

## Production notes

Run behind a real WSGI server, e.g. `gunicorn -w 2 "run:app"`, and generate your own `SECRET_KEY` for the `.env` file (`python -c "import secrets; print(secrets.token_hex(32))"`). Never commit `.env` to a public repository.
Add authentication before exposing it, since CVs contain personal data.
