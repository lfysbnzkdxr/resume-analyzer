"""Evaluation metrics computation."""

import math


def skill_recall(extracted_skills: list[str], expected_skills: list[str]) -> float:
    """What fraction of expected skills were found."""
    if not expected_skills:
        return 1.0
    ext_set = set(s.lower().strip() for s in extracted_skills)
    exp_set = set(s.lower().strip() for s in expected_skills)
    matched = ext_set & exp_set
    return len(matched) / len(exp_set)


def skill_recall_text(raw_text: str, expected_skills: list[str]) -> float:
    """What fraction of expected skills appear as substrings in raw text."""
    if not expected_skills:
        return 1.0
    text_lower = raw_text.lower()
    matched = 0
    for skill in expected_skills:
        if skill.lower() in text_lower:
            matched += 1
    return matched / len(expected_skills)


def score_error(actual_score: float, expected_min: float, expected_max: float) -> float:
    """How far off is the score from the expected range. 0 if within range."""
    if expected_min <= actual_score <= expected_max:
        return 0.0
    if actual_score < expected_min:
        return expected_min - actual_score
    return actual_score - expected_max


def rmse(errors: list[float]) -> float:
    """Root mean squared error."""
    if not errors:
        return 0.0
    mean_sq = sum(e * e for e in errors) / len(errors)
    return math.sqrt(mean_sq)


def compute_report(results: list[dict], expected_skills_per_case: list[list[str]]) -> dict:
    """Aggregate metrics across all test cases.

    Args:
        results: List of dicts from evaluate_single().
        expected_skills_per_case: Parallel list of expected_skills lists,
            used to exclude trivial 1.0 (empty expectation) from avg_skill_recall.
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    score_errors = [r.get("score_error", 0) for r in results if "score_error" in r]
    # Only count recall from cases that actually expect skills
    skills_recalls = [
        r.get("skill_recall", 0)
        for r, exp in zip(results, expected_skills_per_case)
        if "skill_recall" in r and exp
    ]

    return {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_skill_recall": round(sum(skills_recalls) / len(skills_recalls), 4) if skills_recalls else 0,
        "score_rmse": round(rmse(score_errors), 2) if score_errors else None,
    }
