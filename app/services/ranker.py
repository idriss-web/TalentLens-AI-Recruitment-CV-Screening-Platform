from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .embedder import SemanticScorer, explain_top_terms
from .skills import (
    detect_education,
    estimate_experience_years,
    extract_skills,
    find_email,
    guess_name,
    parse_job_description,
    skill_category,
)

COMPONENT_LABELS = {
    "required_skills": "Required skills",
    "semantic": "Semantic fit",
    "preferred_skills": "Preferred skills",
    "experience": "Experience",
    "education": "Education",
}


@dataclass
class Candidate:
    filename: str
    name: str
    email: str | None
    score: float = 0.0
    rank: int = 0
    verdict: str = ""
    components: dict = field(default_factory=dict)
    matched_required: list = field(default_factory=list)
    missing_required: list = field(default_factory=list)
    matched_preferred: list = field(default_factory=list)
    extra_skills: list = field(default_factory=list)
    skills_by_category: dict = field(default_factory=dict)
    experience_years: float = 0.0
    education: str = ""
    shared_terms: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _verdict(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 55:
        return "Good match"
    if score >= 35:
        return "Partial match"
    return "Weak match"


def _ratio(found: int, total: int, empty_value: float = 1.0) -> float:
    return found / total if total else empty_value


class Ranker:
    def __init__(self, scorer: SemanticScorer, weights: dict[str, float]):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.scorer = scorer

    def rank(self, job_text: str, cvs: list[tuple[str, str]]) -> dict:
        job = parse_job_description(job_text)

        req_groups = job["required_groups"]
        pref_groups = job["preferred_groups"]
        wanted = {s for g in req_groups + pref_groups for s in g}

        semantic_scores = self.scorer.score(job_text, [text for _, text in cvs])

        candidates: list[Candidate] = []
        for (filename, text), sem in zip(cvs, semantic_scores):
            skills = extract_skills(text)
            years = estimate_experience_years(text)
            edu_rank, edu_label = detect_education(text)

            def label(g):
                return " / ".join(g)

            matched_req = [label(g) for g in req_groups if any(s in skills for s in g)]
            missing_req = [label(g) for g in req_groups if not any(s in skills for s in g)]
            matched_pref = [label(g) for g in pref_groups if any(s in skills for s in g)]
            extras = sorted((s for s in skills if s not in wanted),
                            key=lambda s: -skills[s])[:10]

            req_score = _ratio(len(matched_req), len(req_groups), empty_value=sem)
            pref_score = _ratio(len(matched_pref), len(pref_groups), empty_value=0.5)
            if job["min_years"] > 0:
                exp_score = min(1.0, years / job["min_years"])
            else:
                exp_score = min(1.0, years / 5.0)
            if job["min_education_rank"] > 0:
                edu_score = min(1.0, edu_rank / job["min_education_rank"])
            else:
                edu_score = min(1.0, edu_rank / 3.0)

            raw = {
                "required_skills": req_score,
                "semantic": sem,
                "preferred_skills": pref_score,
                "experience": exp_score,
                "education": edu_score,
            }
            components = {
                k: {
                    "label": COMPONENT_LABELS[k],
                    "score": round(v * 100, 1),
                    "weight": round(self.weights[k] * 100, 1),
                    "points": round(v * self.weights[k] * 100, 1),
                }
                for k, v in raw.items()
            }
            total = round(sum(c["points"] for c in components.values()), 1)

            cand = Candidate(
                filename=filename,
                name=guess_name(text, fallback=filename.rsplit(".", 1)[0]),
                email=find_email(text),
                score=total,
                verdict=_verdict(total),
                components=components,
                matched_required=matched_req,
                missing_required=missing_req,
                matched_preferred=matched_pref,
                extra_skills=extras,
                experience_years=years,
                education=edu_label,
                shared_terms=explain_top_terms(job_text, text),
            )
            by_cat: dict[str, list[str]] = {}
            for s in sorted(skills, key=lambda s: -skills[s]):
                by_cat.setdefault(skill_category(s), []).append(s)
            cand.skills_by_category = by_cat
            cand.reasons, cand.warnings = self._explain(cand, job, len(req_groups), len(pref_groups))
            candidates.append(cand)

        candidates.sort(key=lambda c: (-c.score, -len(c.matched_required), c.name))
        for i, c in enumerate(candidates, start=1):
            c.rank = i

        return {
            "job": job,
            "backend": self.scorer.backend,
            "backend_note": self.scorer.note,
            "weights": {COMPONENT_LABELS[k]: round(v * 100, 1) for k, v in self.weights.items()},
            "candidates": [c.to_dict() for c in candidates],
        }

    @staticmethod
    def _explain(c: Candidate, job: dict, n_req: int, n_pref: int) -> tuple[list[str], list[str]]:
        reasons, warnings = [], []

        if n_req:
            reasons.append(
                f"Covers {len(c.matched_required)} of {n_req} required skills"
                + (f" ({', '.join(c.matched_required[:6])}{'…' if len(c.matched_required) > 6 else ''})."
                   if c.matched_required else ".")
            )
        if c.missing_required:
            warnings.append("Missing required: " + ", ".join(c.missing_required) + ".")
        if n_pref and c.matched_preferred:
            reasons.append(f"Also has {len(c.matched_preferred)} of {n_pref} nice-to-have skills "
                           f"({', '.join(c.matched_preferred[:5])}).")

        min_years = job["min_years"]
        if min_years:
            if c.experience_years >= min_years:
                reasons.append(f"~{c.experience_years:g} years of experience meets the {min_years:g}+ year requirement.")
            else:
                warnings.append(f"~{c.experience_years:g} years of experience is below the {min_years:g}+ year requirement.")
        elif c.experience_years:
            reasons.append(f"~{c.experience_years:g} years of experience detected.")
        else:
            warnings.append("Could not detect years of experience (no date ranges or 'N years' statement).")

        if c.education != "Not specified":
            reasons.append(f"Highest education detected: {c.education}.")

        sem = c.components["semantic"]["score"]
        if sem >= 60:
            reasons.append(f"CV content is semantically close to the job description ({sem:.0f}/100).")
        elif sem < 30:
            warnings.append(f"CV content is only loosely related to the job description ({sem:.0f}/100).")

        if c.extra_skills:
            reasons.append("Additional relevant skills: " + ", ".join(c.extra_skills[:5]) + ".")
        return reasons, warnings
