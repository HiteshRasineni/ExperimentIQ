"""End-to-end RAG pipeline for ExperimentIQ.

Integrates structured (MLflow API) and unstructured (vector)
retrieval with an LLM answer generation step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Import MLflow client lazily to avoid hard dependency at import time
# The MLflow-related classes are only needed in the constructor
from .router import route_query, parse_route_flag
from .schemas import (
    AskRequest,
    AskResponse,
    SourceDocument,
    StructuredResult,
    UnstructuredResult,
    PipelineResult,
)
from .structured.client import MLflowClientWrapper as StructuredClient
from .structured.query import StructuredQueryExecutor
from .unstructured.client import QdrantClientWrapper


class RAGPipeline:
    """End-to-end RAG pipeline for ExperimentIQ."""

    def __init__(
        self,
        mlflow_client: Optional[StructuredClient] = None,
        qdrant_client: Optional[QdrantClientWrapper] = None,
        llm_provider: Optional[str] = None,
        tracking_uri: str = "http://127.0.0.1:5000",
    ):
        import os
        uri = os.environ.get("MLFLOW_TRACKING_URI", tracking_uri)
        self.mlflow_client = mlflow_client or StructuredClient(tracking_uri=uri)
        self.qdrant_client = qdrant_client or QdrantClientWrapper()
        self.structured_query = StructuredQueryExecutor(self.mlflow_client)
        self.llm_provider = llm_provider
        self._initialized = False

    def initialize(self):
        """Ensure all backends are ready."""
        if not self._initialized:
            self.qdrant_client._ensure_collection()
            self._initialized = True

    def _structured_results(self, query_log: str) -> StructuredResult:
        """Format structured results as SourceDocuments."""
        results = []
        results.append(SourceDocument(
            id=f"run_{query_log[:8]}",
            title="MLflow Run",
            snippet=f"Structured query executed: {query_log[:100]}",
            score=0.9,
            metadata={"query": query_log},
        ))
        return results

    def _unstructured_results(self, query: str, top_k: int) -> List[SourceDocument]:
        """Query Qdrant for unstructured documents."""
        results = self.qdrant_client.query(
            query_text=query,
            top_k=top_k,
        )
        sources = []
        for r in results:
            sources.append(SourceDocument(
                type=r.get("type", "log"),
                id=r["id"],
                title=r.get("title", "Untitled"),
                snippet=r["text"][:500],
                score=min(1.0, max(0.0, float(r.get("score", 0.5)))),
                metadata=r.get("metadata", {}),
            ))
        return sources

    def answer(self, query: str, route: str = "auto", top_k: int = 5, llm_params: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the full RAG pipeline."""
        self.initialize()

        flag = parse_route_flag(route)
        if flag == "auto":
            decision = route_query(query)
            eff_route = decision.path
            confidence = decision.confidence
        else:
            eff_route = flag
            confidence = 1.0

        sources = []
        structured_res: Optional[StructuredResult] = None
        unstructured_res: Optional[UnstructuredResult] = None

        if eff_route == "structured":
            structured_res = self._execute_structured_query(query)
            sources = self._structured_results_to_sources(structured_res)
        elif eff_route == "unstructured":
            unstructured_res = UnstructuredResult(
                query_type="general_retrieval",
                documents=self._unstructured_results(query, top_k),
                query_log=query,
            )
            sources = self._unstructured_results(query, top_k)
        else:
            structured_res = self._execute_structured_query(query)
            unstructured_res = UnstructuredResult(
                query_type="general_retrieval",
                documents=self._unstructured_results(query, top_k),
                query_log=query,
            )
            sources = self._structured_results_to_sources(structured_res)
            sources.extend(self._unstructured_results(query, top_k))

        answer = self._generate_answer(query, sources)

        return PipelineResult(
            answer=answer,
            route=eff_route,
            structured_results=structured_res,
            unstructured_results=unstructured_res,
            sources=sources,
            confidence=confidence,
            raw={"query": query, "route": eff_route, "sources": len(sources)},
        )

    def _default_experiment_ids(self) -> List[str]:
        try:
            exps = self.mlflow_client.client.search_experiments()
            return [e.experiment_id for e in exps if e.name != "Default"] or [e.experiment_id for e in exps]
        except Exception:
            return ["0"]

    def _norm_metric(self, name: str) -> str:
        n = name.strip().lower().rstrip("?").strip()
        aliases = {
            "validation nll": "val_nll", "validation loss": "val_nll",
            "valid nll": "val_nll", "nll": "val_nll", "loss": "val_nll",
            "accuracy": "accuracy", "acc": "accuracy",
        }
        return aliases.get(n, n.replace(" ", "_"))

    def _execute_structured_query(self, query: str) -> StructuredResult:
        """Parse NL question into a real MLflow query."""
        import re
        q = query.lower()
        exp_ids = self._default_experiment_ids()
        m = re.search(r"best\s+(.+)", q)
        if m:
            metric = self._norm_metric(m.group(1))
            asc = True if metric in ("val_nll",) else True
            if any(w in q for w in ["highest", "max", "largest"]):
                asc = False
            return self.structured_query.query_best_metric(
                experiment_ids=exp_ids, metric_key=metric,
                ascending=asc, max_results=5,
            )
        m2 = re.search(r"(val_nll|accuracy|[\w_]+)\s*(<|>|<=|>=)\s*([\d.]+)", q)
        if m2:
            key, op, val = m2.group(1), m2.group(2), float(m2.group(3))
            key = self._norm_metric(key)
            lo = hi = None
            if op in (">", ">="):
                lo = val
            else:
                hi = val
            return self.structured_query.query_runs_by_metric_range(
                experiment_ids=exp_ids, metric_key=key,
                min_val=lo, max_val=hi, max_results=20,
            )
        m3 = re.search(r"(?:config|dataset|flow_name|flow_config)\s*[=:]\s*([\w.\-]+)", q)
        if m3:
            for pk in ("flow_config", "dataset", "flow_name"):
                if pk in q:
                    return self.structured_query.query_runs_by_param(
                        experiment_ids=exp_ids, param_key=pk,
                        param_value=m3.group(1), max_results=20,
                    )
        # default: best val_nll
        return self.structured_query.query_best_metric(
            experiment_ids=exp_ids, metric_key="val_nll",
            ascending=True, max_results=5,
        )

    def _structured_results_to_sources(self, structured_res: StructuredResult) -> List[SourceDocument]:
        """Convert StructuredResult to SourceDocument list."""
        sources = []
        for r in structured_res.results:
            mv = {k: (v.value if hasattr(v, "value") else v.get("value")) for k, v in (r.metrics or {}).items()}
            nll = mv.get("val_nll", "?")
            acc = mv.get("accuracy", "?")
            cfg = (r.params or {}).get("flow_config", "?")
            ds = (r.params or {}).get("dataset", "?")
            sources.append(SourceDocument(
                type="run",
                id=r.run_id,
                title=f"Run {r.run_id[:8]} ({cfg} / {ds})",
                snippet=f"val_nll={nll} accuracy={acc} config={cfg} dataset={ds}",
                score=0.9,
                metadata={"experiment_id": r.experiment_id, "metrics": mv, "params": r.params},
            ))
        return sources

    def _generate_answer(self, query: str, sources: List[SourceDocument]) -> str:
        """Generate an answer from the sources."""
        if not sources:
            return "I couldn't find any matching experiments. Try rephrasing your question or check that experiments exist in your MLflow tracking server."
        q = query.lower()
        if "best" in q and any(k in q for k in ("nll", "loss", "accuracy", "metric")):
            best = sources[0]
            md = best.metadata or {}
            mv = md.get("metrics", {})
            ps = md.get("params", {})
            return (
                f"Best run for your query is {best.id[:8]} "
                f"(config={ps.get('flow_config', '?')}, dataset={ps.get('dataset', '?')}, "
                f"val_nll={mv.get('val_nll', '?')}, accuracy={mv.get('accuracy', '?')}). "
                f"Retrieved {len(sources)} runs, ordered by the requested metric."
            )
        snippets = [s.snippet for s in sources[:3] if s.snippet]
        if snippets:
            return f"Based on {len(sources)} retrieved sources: " + "; ".join(snippets[:2])
        return f"Found {len(sources)} sources related to your question."
