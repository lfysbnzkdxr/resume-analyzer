"""Analysis orchestrator — coordinates the multi-agent workflow."""

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION

from src.agents.resume_agent import extract_resume, extract_resume_from_text
from src.agents.jd_agent import analyze_jd
from src.agents.matching_agent import evaluate_match
from src.rag.library import add_resume_to_library
from src.core.models import AnalysisResult, DimensionScore, Suggestion


def _extract_resume_timed(pdf_path: str) -> tuple:
    """Run resume extraction with timing. Returns (data, elapsed_ms)."""
    t0 = time.time()
    data = extract_resume(pdf_path)
    elapsed = round((time.time() - t0) * 1000)
    return data, elapsed


def _extract_resume_from_text_timed(resume_text: str) -> tuple:
    """Run resume extraction from text with timing. Returns (data, elapsed_ms)."""
    t0 = time.time()
    data = extract_resume_from_text(resume_text)
    elapsed = round((time.time() - t0) * 1000)
    return data, elapsed


def _analyze_jd_timed(jd_text: str) -> tuple:
    """Run JD analysis with timing. Returns (data, elapsed_ms)."""
    t0 = time.time()
    data = analyze_jd(jd_text)
    elapsed = round((time.time() - t0) * 1000)
    return data, elapsed


def run_single_analysis(pdf_path: str, jd_text: str) -> AnalysisResult:
    """Run a single resume-JD analysis (resume agent ∥ JD agent → matching agent)."""
    timings = {}

    # Step 1+2: Resume extraction and JD analysis run in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        resume_future = executor.submit(_extract_resume_timed, pdf_path)
        jd_future = executor.submit(_analyze_jd_timed, jd_text)

        done, _ = wait(
            [resume_future, jd_future],
            return_when=FIRST_EXCEPTION,
        )
        # If any future raised, propagate the exception and cancel the other
        for f in done:
            exc = f.exception()
            if exc is not None:
                resume_future.cancel()
                jd_future.cancel()
                raise RuntimeError(
                    f"Agent execution failed: {'resume' if f is resume_future else 'jd'} agent error"
                ) from exc

        resume_data, timings["resume_agent_ms"] = resume_future.result()
        jd_data, timings["jd_agent_ms"] = jd_future.result()

    resume_json_str = json.dumps(resume_data, ensure_ascii=False, indent=2)
    jd_json_str = json.dumps(jd_data, ensure_ascii=False, indent=2)

    # Step 3: Evaluate match
    t0 = time.time()
    match_result = evaluate_match(resume_json_str, jd_json_str)
    timings["matching_agent_ms"] = round((time.time() - t0) * 1000)
    timings["total_ms"] = sum(timings.values())

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

    filename = Path(pdf_path).name

    return AnalysisResult(
        resume_filename=filename,
        jd_text=jd_text[:200],
        overall_score=match_result.get("overall_score", 0),
        dimensions=dims,
        suggestions=suggs,
        summary=match_result.get("summary", ""),
        timing_ms=timings,
    )


def run_text_analysis(resume_text: str, resume_filename: str, jd_text: str) -> AnalysisResult:
    """Analyze resume text + JD text directly, without a PDF file.

    Runs extract_resume_from_text (text → structured) in parallel with
    analyze_jd, then evaluates the match.
    """
    timings = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        resume_future = executor.submit(_extract_resume_from_text_timed, resume_text)
        jd_future = executor.submit(_analyze_jd_timed, jd_text)

        done, _ = wait(
            [resume_future, jd_future],
            return_when=FIRST_EXCEPTION,
        )
        for f in done:
            exc = f.exception()
            if exc is not None:
                resume_future.cancel()
                jd_future.cancel()
                raise RuntimeError(
                    f"Agent execution failed: {'resume' if f is resume_future else 'jd'} agent error"
                ) from exc

        resume_data, timings["resume_agent_ms"] = resume_future.result()
        jd_data, timings["jd_agent_ms"] = jd_future.result()

    resume_json_str = json.dumps(resume_data, ensure_ascii=False, indent=2)
    jd_json_str = json.dumps(jd_data, ensure_ascii=False, indent=2)

    t0 = time.time()
    match_result = evaluate_match(resume_json_str, jd_json_str)
    timings["matching_agent_ms"] = round((time.time() - t0) * 1000)
    timings["total_ms"] = sum(timings.values())

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

    return AnalysisResult(
        resume_filename=resume_filename,
        jd_text=jd_text[:200],
        overall_score=match_result.get("overall_score", 0),
        dimensions=dims,
        suggestions=suggs,
        summary=match_result.get("summary", ""),
        timing_ms=timings,
    )


def run_library_add(pdf_path: str) -> dict:
    """Add a resume to the library."""
    return add_resume_to_library(pdf_path)
