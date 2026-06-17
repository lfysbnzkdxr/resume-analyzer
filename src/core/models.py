"""Pydantic models for the resume analyzer."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class Skill(BaseModel):
    name: str
    level: str = "unknown"          # proficient / familiar / basic
    importance: str = "unknown"     # must / plus


class Experience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = []


class Education(BaseModel):
    school: str = ""
    degree: str = ""
    major: str = ""


class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = []


class ResumeInfo(BaseModel):
    """Structured information extracted from a resume PDF."""
    personal_info: dict = {}
    summary: str = ""
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    projects: list[Project] = []


class JDRequirements(BaseModel):
    """Structured information extracted from a job description."""
    role_name: str = ""
    required_skills: list[Skill] = []
    preferred_skills: list[Skill] = []
    experience_required: str = ""
    education_required: str = ""
    responsibilities: list[str] = []


class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    details: str = ""


class Suggestion(BaseModel):
    category: str = ""   # skill_gap / presentation / format / experience
    priority: str = ""   # high / medium / low
    content: str = ""


class AnalysisResult(BaseModel):
    """Full analysis result combining resume and JD."""
    resume_filename: str = ""
    jd_text: str = ""
    overall_score: float = Field(ge=0, le=100, default=0)
    dimensions: list[DimensionScore] = []
    suggestions: list[Suggestion] = []
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
