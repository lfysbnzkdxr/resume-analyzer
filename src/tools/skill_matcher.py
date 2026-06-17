"""Skill overlap calculation tool."""

from langchain_core.tools import tool


@tool
def calculate_skill_overlap(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """Calculate overlap between resume skills and JD-required skills."""
    resume_set = set(s.lower().strip() for s in resume_skills)
    jd_set = set(s.lower().strip() for s in jd_skills)

    matched = resume_set & jd_set
    missing = jd_set - resume_set
    extra = resume_set - jd_set

    overlap_rate = len(matched) / len(jd_set) if jd_set else 0

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),
        "overlap_rate": round(overlap_rate, 4),
        "match_count": len(matched),
        "total_required": len(jd_set),
    }
