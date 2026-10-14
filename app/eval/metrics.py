"""Evaluation metrics for ExperimentIQ.

Calculates accuracy, routing accuracy, and hallucination rate
from benchmark evaluation results.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .benchmark import QuestionResult


def calculate_metrics(results: List[QuestionResult]) -> Dict[str, Any]:
    """Calculate evaluation metrics from results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "route_accuracy": 0.0, "answer_accuracy": 0.0, "hallucination_rate": 0.0}

    route_correct = sum(1 for r in results if r.route_correct)
    answer_correct = sum(1 for r in results if r.answer_correct)
    hallucination_count = sum(1 for r in results if r.hallucination_detected)

    return {
        "total_questions": total,
        "route_accuracy": route_correct / total,
        "answer_accuracy": answer_correct / total,
        "hallucination_rate": hallucination_count / total,
        "route_correct_count": route_correct,
        "answer_correct_count": answer_correct,
        "hallucination_count": hallucination_count,
    }


def category_breakdown(results: List[QuestionResult]) -> Dict[str, Dict[str, float]]:
    """Calculate accuracy per category."""
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "hallucinations": 0}
        categories[cat]["total"] += 1
        if r.answer_correct:
            categories[cat]["correct"] += 1
        if r.hallucination_detected:
            categories[cat]["hallucinations"] += 1

    return {
        cat: {
            "total": v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
            "hallucination_rate": v["hallucinations"] / v["total"] if v["total"] > 0 else 0,
        }
        for cat, v in categories.items()
    }


def format_report(metrics: Dict[str, Any], breakdown: Dict[str, Dict[str, float]]) -> str:
    """Format a human-readable evaluation report."""
    lines = [
        "=" * 50,
        "ExperimentIQ Evaluation Report",
        "=" * 50,
        f"Total questions: {metrics['total_questions']}",
        f"Route accuracy: {metrics['route_accuracy']:.1%}",
        f"Answer accuracy: {metrics['answer_accuracy']:.1%}",
        f"Hallucination rate: {metrics['hallucination_rate']:.1%}",
        "",
        "Category breakdown:",
    ]
    for cat, stats in breakdown.items():
        lines.append(
            f"  {cat:20s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})"
        )
    lines.append("=" * 50)
    return "\n".join(lines)
