"""Analysis orchestrator — coordinates the multi-agent workflow."""

import json

from src.agents.resume_agent import extract_resume
from src.agents.jd_agent import analyze_jd
from src.agents.matching_agent import evaluate_match
from src.agents.library_agent import add_resume_to_library
from src.core.models import AnalysisResult, DimensionScore, Suggestion


def run_single_analysis(pdf_path: str, jd_text: str) -> AnalysisResult:
    """Run a single resume-JD analysis (resume agent → JD agent → matching agent)."""
    # Step 1: Extract resume info
    resume_data = extract_resume(pdf_path)
    resume_json_str = json.dumps(resume_data, ensure_ascii=False, indent=2)

    # Step 2: Analyze JD
    jd_data = analyze_jd(jd_text)
    jd_json_str = json.dumps(jd_data, ensure_ascii=False, indent=2)

    # Step 3: Evaluate match
    match_result = evaluate_match(resume_json_str, jd_json_str)

    # Step 4: Build result
    dims = []
    for d in match_result.get("dimensions", []):
        dims.append(DimensionScore(
            name=d.get("name", ""),
            score=d.get("score", 0),
            weight=d.get("weight", 0),
            details=d.get("details", ""),
        ))

    suggs = []
    for s in match_result.get("suggestions", []):
        suggs.append(Suggestion(
            category=s.get("category", ""),
            priority=s.get("priority", ""),
            content=s.get("content", ""),
        ))

    filename = pdf_path.replace("\\", "/").split("/")[-1]

    return AnalysisResult(
        resume_filename=filename,
        jd_text=jd_text[:200],
        overall_score=match_result.get("overall_score", 0),
        dimensions=dims,
        suggestions=suggs,
        summary=match_result.get("summary", ""),
    )


def run_library_search(jd_text: str) -> dict:
    """Search resume library for best matches to a JD."""
    from src.tools.search import search_resumes_for_jd

    results = search_resumes_for_jd(jd_text)
    return {
        "results": results,
        "total": len(results),
    }


def run_library_add(pdf_path: str) -> dict:
    """Add a resume to the library."""
    return add_resume_to_library(pdf_path)
