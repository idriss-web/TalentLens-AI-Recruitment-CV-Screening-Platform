from __future__ import annotations

import re
from datetime import date

SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    "Python": ("Languages", ["python", "python3"]),
    "SQL": ("Languages", ["sql", "postgresql", "postgres", "mysql"]),
    "Java": ("Languages", ["java"]),
    "C++": ("Languages", ["c++"]),
    "JavaScript": ("Languages", ["javascript", "typescript", "node.js"]),
    "Rust": ("Languages", ["rust"]),
    "Machine Learning": ("ML", ["machine learning", "ml"]),
    "Deep Learning": ("ML", ["deep learning"]),
    "PyTorch": ("ML", ["pytorch", "torch"]),
    "TensorFlow": ("ML", ["tensorflow", "keras"]),
    "scikit-learn": ("ML", ["scikit-learn", "sklearn", "scikit learn"]),
    "XGBoost": ("ML", ["xgboost", "lightgbm"]),
    "Pandas": ("ML", ["pandas"]),
    "NumPy": ("ML", ["numpy"]),
    "Reinforcement Learning": ("ML", ["reinforcement learning", "rlhf"]),
    "Computer Vision": ("ML", ["computer vision", "opencv", "image classification"]),
    "NLP": ("NLP", ["nlp", "natural language processing", "computational linguistics"]),
    "Transformers": ("NLP", ["transformers", "transformer models", "huggingface transformers"]),
    "Hugging Face": ("NLP", ["hugging face", "huggingface"]),
    "BERT": ("NLP", ["bert", "roberta", "distilbert", "deberta"]),
    "LLMs": ("NLP", ["llm", "llms", "large language model", "large language models", "gpt", "llama", "mistral"]),
    "Fine-tuning": ("NLP", ["fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft"]),
    "Prompt Engineering": ("NLP", ["prompt engineering", "prompt design"]),
    "RAG": ("NLP", ["rag", "retrieval augmented generation", "retrieval-augmented generation"]),
    "LangChain": ("NLP", ["langchain", "llamaindex", "llama-index"]),
    "spaCy": ("NLP", ["spacy"]),
    "NLTK": ("NLP", ["nltk"]),
    "Named Entity Recognition": ("NLP", ["ner", "named entity recognition", "entity extraction"]),
    "Text Classification": ("NLP", ["text classification", "sentiment analysis", "intent classification"]),
    "Semantic Search": ("NLP", ["semantic search", "embeddings", "sentence embeddings", "sentence-transformers", "vector search"]),
    "Summarization": ("NLP", ["summarization", "summarisation", "text summarization"]),
    "Machine Translation": ("NLP", ["machine translation", "seq2seq"]),
    "Speech/ASR": ("NLP", ["speech recognition", "asr", "whisper"]),
    "FastAPI": ("Backend", ["fastapi"]),
    "Flask": ("Backend", ["flask"]),
    "Django": ("Backend", ["django"]),
    "REST APIs": ("Backend", ["rest api", "rest apis", "restful", "api development"]),
    "Docker": ("MLOps", ["docker", "containers", "containerization"]),
    "Kubernetes": ("MLOps", ["kubernetes", "k8s"]),
    "MLOps": ("MLOps", ["mlops", "mlflow", "kubeflow", "model deployment", "model serving"]),
    "CI/CD": ("MLOps", ["ci/cd", "github actions", "jenkins", "gitlab ci"]),
    "Git": ("MLOps", ["git", "github", "gitlab"]),
    "AWS": ("Cloud", ["aws", "amazon web services", "sagemaker"]),
    "GCP": ("Cloud", ["gcp", "google cloud", "vertex ai"]),
    "Azure": ("Cloud", ["azure"]),
    "Spark": ("Data", ["spark", "pyspark", "databricks"]),
    "Vector Databases": ("Data", ["faiss", "pinecone", "weaviate", "milvus", "chroma", "pgvector", "qdrant", "vector database", "vector databases"]),
    "Elasticsearch": ("Data", ["elasticsearch", "opensearch"]),
    "MongoDB": ("Data", ["mongodb"]),
    "Airflow": ("Data", ["airflow"]),
    "Model Evaluation": ("Practices", ["model evaluation", "a/b testing", "evaluation metrics", "benchmarking"]),
    "Data Annotation": ("Practices", ["data annotation", "labeling", "labelling", "label studio"]),
    "Research/Publications": ("Practices", ["publications", "published papers", "arxiv", "acl", "emnlp", "neurips"]),
}

_ALIAS_TO_SKILL: dict[str, str] = {}
for _skill, (_cat, _aliases) in SKILL_TAXONOMY.items():
    _ALIAS_TO_SKILL[_skill.lower()] = _skill
    for _a in _aliases:
        _ALIAS_TO_SKILL[_a.lower()] = _skill


def _build_pattern(alias: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#])" + re.escape(alias) + r"(?![\w+#])", re.IGNORECASE)


_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_build_pattern(alias), skill)
    for alias, skill in sorted(_ALIAS_TO_SKILL.items(), key=lambda kv: -len(kv[0]))
]

_CASE_SENSITIVE = {"ml", "ner", "rag", "asr", "acl", "gpt", "llm", "llms", "lora", "aws", "gcp", "nlp", "sql", "git"}


def extract_skill_spans(text: str) -> list[tuple[int, int, str]]:
    taken: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []
    for pattern, skill in _ALIAS_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0)
            if token.lower() in _CASE_SENSITIVE and token == token.lower():
                continue
            s, e = match.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            spans.append((s, e, skill))
    spans.sort()
    return spans


def extract_skills(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for _, _, skill in extract_skill_spans(text):
        found[skill] = found.get(skill, 0) + 1
    return found


_OR_GAP = re.compile(r"\s*(,\s*)?(or|/)\s*", re.IGNORECASE)
_COMMA_GAP = re.compile(r"\s*,\s*")


def extract_skill_groups(line: str) -> list[tuple[str, ...]]:
    spans = extract_skill_spans(line)
    n = len(spans)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    gaps = [line[spans[i][1]:spans[i + 1][0]] for i in range(n - 1)]
    in_or = [False] * n
    for i, gap in enumerate(gaps):
        if _OR_GAP.fullmatch(gap):
            union(i, i + 1)
            in_or[i] = in_or[i + 1] = True
    changed = True
    while changed:
        changed = False
        for i, gap in enumerate(gaps):
            if (_COMMA_GAP.fullmatch(gap) and find(i) != find(i + 1) and (in_or[i] or in_or[i + 1])
                    and skill_category(spans[i][2]) == skill_category(spans[i + 1][2])):
                union(i, i + 1)
                in_or[i] = in_or[i + 1] = True
                changed = True

    groups: dict[int, list[str]] = {}
    for i, (_, _, skill) in enumerate(spans):
        members = groups.setdefault(find(i), [])
        if skill not in members:
            members.append(skill)
    return [tuple(sorted(m)) for m in groups.values()]


def skill_category(skill: str) -> str:
    return SKILL_TAXONOMY.get(skill, ("Other", []))[0]


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

_EXPLICIT_YEARS = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|professional|industry|hands-on|work))?\s*(?:experience|exp)?",
    re.IGNORECASE,
)
_DATE_RANGE = re.compile(
    r"(?:(?P<m1>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y1>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
    r"(?:(?P<present>present|current|now|ongoing)|(?:(?P<m2>[A-Za-z]{3})[a-z]*\.?\s+)?(?P<y2>(?:19|20)\d{2}))",
    re.IGNORECASE,
)


def _month_index(year: int, mon: str | None, default: int) -> int:
    m = _MONTHS.get((mon or "").lower()[:3], default)
    return year * 12 + m


def estimate_experience_years(text: str, today: date | None = None) -> float:
    today = today or date.today()
    now_idx = today.year * 12 + today.month

    intervals: list[tuple[int, int]] = []
    for m in _DATE_RANGE.finditer(text):
        start = _month_index(int(m.group("y1")), m.group("m1"), 1)
        if m.group("present"):
            end = now_idx
        else:
            end = _month_index(int(m.group("y2")), m.group("m2"), 12) + 1
        if end >= start and (end - start) <= 12 * 40:
            intervals.append((start, end))

    intervals.sort()
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    from_dates = sum(e - s for s, e in merged) / 12.0

    claims = [int(x) for x in _EXPLICIT_YEARS.findall(text) if 0 < int(x) <= 40]
    from_claims = float(max(claims)) if claims else 0.0

    return round(min(40.0, max(from_dates, from_claims)), 1)


_EDU_LEVELS = [
    (4, "PhD", re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I)),
    (3, "Master's", re.compile(r"\b(m\.?sc|m\.?s\.?|master'?s?|m\.?eng|mba|m\.?tech)\b", re.I)),
    (2, "Bachelor's", re.compile(r"\b(b\.?sc|b\.?s\.?|bachelor'?s?|b\.?eng|b\.?tech|licence|licenc[ei]e)\b", re.I)),
    (1, "Diploma/Bootcamp", re.compile(r"\b(diploma|associate|bootcamp|certificate)\b", re.I)),
]


def detect_education(text: str) -> tuple[int, str]:
    for rank, label, pattern in _EDU_LEVELS:
        if pattern.search(text):
            return rank, label
    return 0, "Not specified"


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def guess_name(text: str, fallback: str) -> str:
    for line in text.splitlines()[:6]:
        line = line.strip()
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isalpha() and w[0].isupper() for w in words):
            return line
    return fallback


def find_email(text: str) -> str | None:
    m = _EMAIL.search(text)
    return m.group(0) if m else None


_PREFERRED_HEADINGS = re.compile(
    r"(nice[- ]to[- ]have|preferred|bonus|plus|good to have|desirable|a plus|advantageous)", re.I)
_REQUIRED_HEADINGS = re.compile(
    r"(requirements?|must[- ]have|required|qualifications?|what you('ll)? need|you have|skills)", re.I)
_IGNORED_HEADINGS = re.compile(
    r"(about|responsibilit|what you('ll)? do|benefits|we offer|perks|why join|our (team|company)|compensation)", re.I)
_INLINE_PREFERRED = re.compile(r"(nice[- ]to[- ]have|a plus|bonus|preferred)", re.I)


def _collect_groups(text: str, start_mode: str) -> tuple[dict, dict]:
    required: dict[tuple, int] = {}
    preferred: dict[tuple, int] = {}
    mode = start_mode

    for raw in text.splitlines():
        line = raw.strip().lstrip("#*-• ").strip()
        if not line:
            continue
        if len(line) < 60:
            if _PREFERRED_HEADINGS.search(line):
                mode = "preferred"
            elif _REQUIRED_HEADINGS.search(line):
                mode = "required"
            elif _IGNORED_HEADINGS.search(line):
                mode = "ignore"
        if mode == "ignore":
            continue
        inline_preferred = len(line) >= 40 and _INLINE_PREFERRED.search(line)
        target = preferred if (mode == "preferred" or inline_preferred) else required
        for group in extract_skill_groups(line):
            target[group] = target.get(group, 0) + 1
    return required, preferred


def parse_job_description(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    has_sections = any(
        len(l) < 60 and (_REQUIRED_HEADINGS.search(l) or _PREFERRED_HEADINGS.search(l)) for l in lines)

    required, preferred = _collect_groups(text, "ignore" if has_sections else "required")
    if not required:
        required, preferred = _collect_groups(text, "required")

    req_skills = {s for g in required for s in g}
    preferred = {g: c for g, c in preferred.items() if not set(g) <= req_skills}

    claims = [int(x) for x in _EXPLICIT_YEARS.findall(text) if 0 < int(x) <= 20]
    min_years = float(min(claims)) if claims else 0.0
    edu_rank, edu_label = detect_education(text)

    req_groups = sorted([list(g) for g in required])
    pref_groups = sorted([list(g) for g in preferred])
    return {
        "required_groups": req_groups,
        "preferred_groups": pref_groups,
        "required_skills": [" / ".join(g) for g in req_groups],
        "preferred_skills": [" / ".join(g) for g in pref_groups],
        "min_years": min_years,
        "min_education_rank": edu_rank,
        "min_education_label": edu_label,
    }
