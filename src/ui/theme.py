"""Theme constants for the UI layer."""

# Score thresholds and colors
SCORE_COLORS: dict[str, str] = {
    "high": "#22c55e",
    "medium": "#eab308",
    "low": "#ef4444",
}

SCORE_THRESHOLD_HIGH = 85
SCORE_THRESHOLD_MEDIUM = 65


def score_color(score: float) -> str:
    """Map a score (0-100) to a color."""
    if score >= SCORE_THRESHOLD_HIGH:
        return SCORE_COLORS["high"]
    elif score >= SCORE_THRESHOLD_MEDIUM:
        return SCORE_COLORS["medium"]
    return SCORE_COLORS["low"]
