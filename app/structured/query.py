"""Structured query execution for ExperimentIQ.

Transforms natural-language queries into MLflow API calls
and formats results for the RAG pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlflow

from .client import MLflowClientWrapper
from ..schemas import RunSummary, StructuredResult


class StructuredQueryExecutor:
    """Executes structured queries against MLflow tracking data."""

    def __init__(self, mlflow_client: MLflowClientWrapper):
        self.mlflow_client = mlflow_client

    def _run_to_summary(self, run: mlflow.entities.Run) -> RunSummary:
        """Convert an MLflow Run to a RunSummary."""
        params = {k: v for k, v in (run.data.params or {}).items()} if hasattr(run.data, "params") else dict(run.data.params or {})
        # run.data.params may be dict already (mlflow 3.x); handle both
        try:
            params = dict(run.data.params) if run.data.params else {}
        except Exception:
            params = {}
        try:
            tags = dict(run.data.tags) if run.data.tags else {}
        except Exception:
            tags = {}
        metrics = {}
        try:
            for key, val in (run.data.metrics or {}).items():
                metrics[key] = {"key": key, "value": float(val), "epoch": 0}
        except Exception:
            pass
        return RunSummary(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            run_name=run.info.run_name,
            status=run.info.status,
            start_time=run.info.start_time,
            end_time=run.info.end_time,
            params=params,
            metrics=metrics,
            tags=tags,
        )


    def summarise_query(self, query: str, experiments: List[str]) -> StructuredResult:
        """Dispatch a natural language query to the appropriate method."""
        q_lower = query.lower().strip()

        # Pattern: "which X gave the best Y"
        import re
        best_match = re.search(r"best\s+(.+?)\s+$", q_lower)
        if best_match:
            metric = best_match.group(1).strip().rstrip("?").strip()
            experiment_ids = self._resolve_experiment_ids(experiments)
            return self.query_best_metric(
                experiment_ids=experiment_ids,
                metric_key=metric,
                ascending=True,
                max_results=1,
            )

        # Pattern: "how many runs"
        if "how many runs" in q_lower or "count of runs" in q_lower:
            experiments = self._resolve_all_experiments()
            summaries = [RunSummary(
                run_id="count",
                experiment_id="meta",
                run_name="COUNT",
                status="COMPLETED",
                start_time=None,
                end_time=None,
                params={},
                metrics={},
                tags={},
            )]
            return StructuredResult(
                query_type="count",
                results=summaries,
                query_log=f"COUNT(*) FROM runs WHERE experiment_id IN ({','.join(experiments)})",
            )

        # Default: try as metric search
        experiment_ids = self._resolve_experiment_ids(experiments)
        return self.query_best_metric(
            experiment_ids=experiment_ids,
            metric_key="val_nll",
            ascending=True,
            max_results=10,
        )

    def _resolve_experiment_ids(self, experiment_ids: List[str]) -> List[str]:
        """Resolve experiment IDs/names to IDs."""
        return list(experiment_ids)

    def _resolve_all_experiments(self) -> List[str]:
        """Get all experiment IDs."""
        try:
            exps = self.mlflow_client.list_experiments()
            return [e.experiment_id for e in exps]
        except Exception:
            return []


    def query_best_metric(
        self,
        experiment_ids: List[str],
        metric_key: str,
        ascending: bool = True,
        max_results: int = 10,
        filter_string: str = "",
    ) -> StructuredResult:
        """Query for the best runs by a metric."""
        runs = self.mlflow_client.find_best_runs_by_metric(
            experiment_ids=experiment_ids,
            metric_key=metric_key,
            ascending=ascending,
            max_results=max_results,
            filter_string=filter_string,
        )
        summaries = [self._run_to_summary(r) for r in runs]
        return StructuredResult(
            query_type="metric_search",
            results=summaries,
            query_log=f"SELECT * FROM runs WHERE metric={metric_key} ORDER BY {metric_key} {'ASC' if ascending else 'DESC'} LIMIT {max_results}",
        )

    def query_runs_by_metric_range(
        self,
        experiment_ids: List[str],
        metric_key: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        max_results: int = 50,
    ) -> StructuredResult:
        """Query for runs with metric values in a range."""
        runs = self.mlflow_client.get_runs_by_metric_range(
            experiment_ids=experiment_ids,
            metric_key=metric_key,
            min_val=min_val,
            max_val=max_val,
            max_results=max_results,
        )
        summaries = [self._run_to_summary(r) for r in runs]
        return StructuredResult(
            query_type="metric_range",
            results=summaries,
            query_log=f"SELECT * FROM runs WHERE metric={metric_key} BETWEEN {min_val} AND {max_val} LIMIT {max_results}",
        )

    def query_runs_by_param(
        self,
        experiment_ids: List[str],
        param_key: str,
        param_value: str,
        max_results: int = 50,
    ) -> StructuredResult:
        """Query for runs with a specific parameter value."""
        runs = self.mlflow_client.find_runs_with_param_value(
            experiment_ids=experiment_ids,
            param_key=param_key,
            param_value=param_value,
            max_results=max_results,
        )
        summaries = [self._run_to_summary(r) for r in runs]
        return StructuredResult(
            query_type="param_filter",
            results=summaries,
            query_log=f"SELECT * FROM runs WHERE params.{param_key} = '{param_value}' LIMIT {max_results}",
        )

    def query_run_summary(self, run_id: str) -> StructuredResult:
        """Get a summary of a single run."""
        run = self.mlflow_client.get_run(run_id)
        if run is None:
            return StructuredResult(
                query_type="run_summary",
                results=[],
                query_log=f"GET RUN {run_id}",
            )
        summaries = [self._run_to_summary(run)]
        return StructuredResult(
            query_type="run_summary",
            results=summaries,
            query_log=f"GET RUN {run_id}",
        )
