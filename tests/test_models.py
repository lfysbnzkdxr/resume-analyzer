"""Tests for Pydantic model validation."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.core.models import (
    AnalysisResult,
    DimensionScore,
    Education,
    Experience,
    Project,
    ResumeInfo,
    Skill,
    Suggestion,
)


class TestSkill:
    def test_defaults(self):
        s = Skill(name="python")
        assert s.level == "unknown"
        assert s.importance == "unknown"

    def test_full(self):
        s = Skill(name="python", level="proficient", importance="must")
        assert s.level == "proficient"


class TestExperience:
    def test_defaults(self):
        e = Experience()
        assert e.company == ""
        assert e.highlights == []

    def test_with_data(self):
        e = Experience(company="Acme", role="Engineer", highlights=["did stuff"])
        assert e.company == "Acme"
        assert e.highlights == ["did stuff"]


class TestEducation:
    def test_defaults(self):
        edu = Education()
        assert edu.school == ""

    def test_with_data(self):
        edu = Education(school="MIT", degree="BS", major="CS")
        assert edu.degree == "BS"


class TestProject:
    def test_defaults(self):
        p = Project()
        assert p.technologies == []

    def test_with_data(self):
        p = Project(name="My App", technologies=["python", "react"])
        assert len(p.technologies) == 2


class TestResumeInfo:
    def test_defaults(self):
        r = ResumeInfo()
        assert r.skills == []
        assert r.experience == []

    def test_with_data(self):
        r = ResumeInfo(
            personal_info={"name": "Alice"},
            skills=["python"],
            experience=[Experience(company="Co")],
        )
        assert r.personal_info["name"] == "Alice"
        assert len(r.experience) == 1


class TestDimensionScore:
    def test_valid(self):
        d = DimensionScore(name="tech", score=85, weight=0.5, details="good")
        assert d.score == 85

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            DimensionScore(name="tech", score=150, weight=0.5)

    def test_negative_score(self):
        with pytest.raises(ValidationError):
            DimensionScore(name="tech", score=-1, weight=0.5)


class TestSuggestion:
    def test_defaults(self):
        s = Suggestion()
        assert s.category == ""
        assert s.priority == ""

    def test_with_data(self):
        s = Suggestion(category="skill_gap", priority="high", content="learn go")
        assert s.content == "learn go"


class TestAnalysisResult:
    def test_defaults(self):
        r = AnalysisResult()
        assert r.overall_score == 0
        assert isinstance(r.created_at, datetime)

    def test_valid(self):
        r = AnalysisResult(
            overall_score=75,
            dimensions=[DimensionScore(name="tech", score=80, weight=0.5)],
            suggestions=[Suggestion(content="improve")],
            summary="ok",
        )
        assert r.overall_score == 75
        assert len(r.dimensions) == 1

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            AnalysisResult(overall_score=101)
