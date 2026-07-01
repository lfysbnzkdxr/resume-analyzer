"""Tests for skill_matcher — pure function, no mocking needed.

Note: calculate_skill_overlap is a @tool-decorated function, so we call
it via .invoke() with the proper dict format.
"""

from src.tools.skill_matcher import calculate_skill_overlap


def _call(**kwargs):
    """Helper to invoke the LangChain tool correctly."""
    return calculate_skill_overlap.invoke(kwargs)


def test_full_overlap():
    result = _call(resume_skills=["python", "java", "sql"], jd_skills=["python", "java", "sql"])
    assert result["overlap_rate"] == 1.0
    assert len(result["missing_skills"]) == 0
    assert result["match_count"] == 3


def test_partial_overlap():
    result = _call(resume_skills=["python", "java", "docker"], jd_skills=["python", "go", "kubernetes", "rust"])
    assert result["overlap_rate"] == 0.25
    assert "python" in result["matched_skills"]
    assert result["match_count"] == 1
    assert result["total_required"] == 4


def test_no_overlap():
    result = _call(resume_skills=["react", "css"], jd_skills=["python", "java"])
    assert result["overlap_rate"] == 0.0
    assert result["match_count"] == 0


def test_empty_skills():
    result = _call(resume_skills=[], jd_skills=["python", "java"])
    assert result["overlap_rate"] == 0.0
    assert result["match_count"] == 0
    assert len(result["missing_skills"]) == 2


def test_empty_jd_skills():
    result = _call(resume_skills=["python", "java"], jd_skills=[])
    assert result["overlap_rate"] == 0.0
    assert result["total_required"] == 0


def test_case_insensitive():
    result = _call(resume_skills=["Python", "Java"], jd_skills=["python", "java"])
    assert result["match_count"] == 2
    assert result["overlap_rate"] == 1.0


def test_whitespace_trim():
    result = _call(resume_skills=["  python ", "java  "], jd_skills=["python", "java"])
    assert result["match_count"] == 2
