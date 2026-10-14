"""Benchmark dataset and evaluation harness for ExperimentIQ.

Contains hand-authored question-answer pairs and metrics
for measuring answer accuracy, routing accuracy, and hallucination rate.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with its ground truth answer."""

    id: str
    question: str
    expected_route: str
    expected_answer: str
    category: str
    difficulty: str = "easy"


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""

    question_id: str
    expected_route: str
    actual_route: str
    route_correct: bool
    expected_answer: str
    actual_answer: str
    answer_correct: bool
    category: str
    hallucination_detected: bool


def evaluate_question(question: BenchmarkQuestion, actual_route: str, actual_answer: str) -> QuestionResult:
    """Evaluate a single question against the benchmark."""
    route_correct = actual_route == question.expected_route
    answer_correct = question.expected_answer.lower() in actual_answer.lower()

    hallucination_detected = False
    if question.category == "log_retrieval" and not any(word in actual_answer.lower() for word in ["log", "trace", "output"]):
        hallucination_detected = True

    return QuestionResult(
        question_id=question.id,
        expected_route=question.expected_route,
        actual_route=actual_route,
        route_correct=route_correct,
        expected_answer=question.expected_answer,
        actual_answer=actual_answer,
        answer_correct=answer_correct,
        category=question.category,
        hallucination_detected=hallucination_detected,
    )
