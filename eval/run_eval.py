"""Run evaluation against test cases."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from eval.metrics import skill_recall, skill_recall_text, score_error, compute_report
from eval.test_cases import test_cases


def evaluate_single(case: dict) -> dict:
    """Run one test case through the analysis pipeline and return metrics."""
    import tempfile
    from src.core.orchestrator import run_single_analysis

    # Save resume text as a temp file (orchestrator will try to parse as PDF)
    # For eval without real PDFs, we feed text directly
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w", encoding="utf-8")
    tmp.write(case["resume_text"])
    tmp.close()

    try:
        start = time.time()
        result = run_single_analysis(tmp.name, case["jd_text"])
        elapsed = time.time() - start

        # Extract skills by checking if each expected skill name
        # appears as a substring in any dimension's details or the summary.
        # This is more reliable than trying to parse free-form Chinese text.
        detail_text = " ".join(d.details or "" for d in result.dimensions)
        detail_text += " " + (result.summary or "")

        s_recall = skill_recall_text(detail_text, case.get("expected_skills", []))
        s_err = score_error(
            result.overall_score,
            case.get("expected_score_min", 0),
            case.get("expected_score_max", 100),
        )

        passed = s_err == 0
        notes = []
        if s_err > 0:
            notes.append(
                f"评分 {result.overall_score:.0f} 超出期望范围 "
                f"{case.get('expected_score_min', 0)}-{case.get('expected_score_max', 100)}"
            )

        return {
            "id": case["id"],
            "category": case["category"],
            "passed": passed,
            "overall_score": result.overall_score,
            "expected_range": f"{case.get('expected_score_min', 0)}-{case.get('expected_score_max', 100)}",
            "skill_recall": s_recall,
            "score_error": s_err,
            "suggestion_count": len(result.suggestions),
            "response_time_s": round(elapsed, 2),
            "timing_ms": result.timing_ms,
            "notes": "; ".join(notes),
        }
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "error": str(e),
        }
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def run_eval(output_path: str = "eval_report.json"):
    """Run all test cases and save the report."""
    print(f"Running {len(test_cases)} test cases...\n")
    results = []
    for case in test_cases:
        print(f"  [{case['id']}] ({case['category']})... ", end="", flush=True)
        r = evaluate_single(case)
        status = "✅ PASS" if r.get("passed") else "❌ FAIL"
        print(f"{status}")
        if not r.get("passed") and r.get("notes"):
            print(f"    ↳ {r['notes']}")
        results.append(r)

    report = compute_report(results, [c.get("expected_skills", []) for c in test_cases])
    report["results"] = results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"总用例: {report['total_cases']}")
    print(f"通过: {report['passed']} | 失败: {report['failed']}")
    print(f"通过率: {report['pass_rate']:.1%}")
    print(f"技能召回率均值: {report['avg_skill_recall']:.1%}")
    print(f"评分 RMSE: {report['score_rmse']}")

    # Timing summary across all cases
    timings = [r.get("timing_ms", {}) for r in report["results"] if r.get("timing_ms")]
    if timings:
        avg_resume = sum(t.get("resume_agent_ms", 0) for t in timings) / len(timings)
        avg_jd = sum(t.get("jd_agent_ms", 0) for t in timings) / len(timings)
        avg_match = sum(t.get("matching_agent_ms", 0) for t in timings) / len(timings)
        avg_total = sum(t.get("total_ms", 0) for t in timings) / len(timings)
        print(f"耗时均值:")
        print(f"  Resume Agent:      {avg_resume:>6.0f}ms")
        print(f"  JD Agent:          {avg_jd:>6.0f}ms")
        print(f"  Matching Agent:    {avg_match:>6.0f}ms")
        print(f"  ──────────────────────────")
        print(f"  Total:             {avg_total:>6.0f}ms")
    print(f"{'='*50}")
    print(f"报告已保存: {output_path}")

    # Append to history report
    _append_history(report)


def _append_history(report: dict):
    """Append this run's metrics to eval/report.md as a markdown table row."""
    import os
    history_path = Path(__file__).resolve().parent / "report.md"
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Timing
    avg_total_ms = 0
    timings = [r.get("timing_ms", {}) for r in report["results"] if r.get("timing_ms")]
    if timings:
        avg_total_ms = sum(t.get("total_ms", 0) for t in timings) / len(timings)

    new_row = (
        f"| {today} "
        f"| {report['pass_rate']:.1%} "
        f"| {report['total_cases']}/{report['failed']} "
        f"| {report['avg_skill_recall']:.1%} "
        f"| {report['score_rmse']} "
        f"| {avg_total_ms:.0f}ms |\n"
    )

    if not history_path.exists():
        header = (
            "# Eval 历史记录\n\n"
            "每条记录对应一次 `python eval/run_eval.py`。\n\n"
            "| 日期 | 通过率 | 通过/失败 | 技能召回率 | 评分RMSE | 平均总耗时 |\n"
            "|------|--------|-----------|------------|----------|------------|\n"
        )
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(header + new_row)
    else:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(new_row)

    print(f"历史记录已追加: {history_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "eval_report.json"
    run_eval(output)
