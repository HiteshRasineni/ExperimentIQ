import os

content = '''"""Question routing logic for ExperimentIQ."""

from __future__ import annotations
from typing import Optional

STRUCTURED_KEYWORDS = [
    'best', 'highest', 'lowest', 'which', 'most', 'least',
    'minimum', 'maximum', 'compare', 'comparing', 'vs',
    'versus', 'all runs', 'across runs', 'total', 'sum',
    'average', 'mean', 'median', 'count', 'first', 'last',
    'earliest', 'latest', 'validation nll', 'validation loss',
    'val_nll', 'validation', 'metric', 'metrics', 'parameter',
    'params', 'config', 'flow config', 'dataset',
]

UNSTRUCTURED_KEYWORDS = [
    'log', 'logs', 'trace', 'error', 'warning', 'message',
    'note', 'description', 'comment', 'summary', 'overview',
    'report', 'why did', 'what happened',
]

STRUCTURED_PHRASES = [
    'best validation', 'best nll', 'highest validation',
    'lowest validation', 'which run', 'which flow config',
    'across all runs',
]

UNSTRUCTURED_PHRASES = [
    'what do the logs say', 'what are the logs', 'logging info',
    'log messages', 'tell me about', 'explain the configuration',
    'what is in', 'overview of',
]

def _contains_phrase(text: str, phrases: list[str]) -> bool:
    text_lower = text.lower().strip()
    return any(phrase.lower().strip() in text_lower for phrase in phrases)


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


class RouteDecision:
    def __init__(self, path: str, confidence: float, reasons: list[str], original_query: str):
        self.path = path
        self.confidence = confidence
        self.reasons = reasons
        self.original_query = original_query

    def __repr__(self):
        return f'RouteDecision(path={self.path}, confidence={self.confidence:.2f}, reasons={self.reasons})'


def route_query(query: str, llm_fallback: Optional[bool] = None) -> RouteDecision:
    reasons = []
    query_lower = query.lower().strip()

    if _contains_phrase(query_lower, STRUCTURED_PHRASES):
        reasons.append('Matched structured query phrase')
        return RouteDecision(path='structured', confidence=0.95, reasons=reasons, original_query=query)

    if _contains_phrase(query_lower, UNSTRUCTURED_PHRASES):
        reasons.append('Matched unstructured query phrase')
        return RouteDecision(path='unstructured', confidence=0.95, reasons=reasons, original_query=query)

    struct_score = _count_keyword_matches(query_lower, STRUCTURED_KEYWORDS)
    unstruct_score = _count_keyword_matches(query_lower, UNSTRUCTURED_KEYWORDS)

    if struct_score > unstruct_score:
        reasons.append(f'Structured keywords: {struct_score}, Unstructured: {unstruct_score}')
        return RouteDecision(path='structured', confidence=0.7 + 0.05 * struct_score, reasons=reasons, original_query=query)

    if unstruct_score > struct_score:
        reasons.append(f'Unstructured keywords: {unstruct_score}, Structured: {struct_score}')
        return RouteDecision(path='unstructured', confidence=0.7 + 0.05 * unstruct_score, reasons=reasons, original_query=query)

    if llm_fallback:
        reasons.append('Tie - using LLM fallback')
        return _llm_route_fallback(query_lower, reasons)

    reasons.append('Tie, no strong signal - defaulting to structured')
    return RouteDecision(path='structured', confidence=0.5, reasons=reasons, original_query=query)


def _llm_route_fallback(query_lower: str, reasons: list[str]) -> RouteDecision:
    if any(word in query_lower for word in ['why', 'explain', 'summary', 'story', 'history']):
        reasons.append('LLM fallback: qualitative intent detected')
        return RouteDecision(path='unstructured', confidence=0.8, reasons=reasons, original_query=query_lower)
    reasons.append('LLM fallback: quantitative intent detected')
    return RouteDecision(path='structured', confidence=0.8, reasons=reasons, original_query=query_lower)


def parse_route_flag(route: Optional[str]) -> str:
    if route is None:
        return 'auto'
    route = route.lower()
    if route in ('structured', 'unstructured'):
        return route
    if route in ('auto', 'default'):
        return 'auto'
    aliases = {
        'api': 'structured', 'mlflow': 'structured',
        'vector': 'unstructured', 'vdb': 'unstructured',
        'log': 'unstructured', 'search': 'structured',
    }
    return aliases.get(route, 'auto')
'''

with open(r'C:/Users/rshit/Downloads/ExperimentIQ/app/router.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('router.py written successfully')
print(f'Length: {len(content)} chars')
