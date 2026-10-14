"""MLflow Tracking API client wrapper.

Provides a clean interface to the MLflow Tracking Server for
experiment, run, metric, and parameter queries.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from typing import Any, Dict, List, Optional


class MLflowClientWrapper:
    """Wrapper around MLflowClient with ExperimentIQ-friendly methods."""

    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        run_view_type: int = 1,
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
        output_filter: Optional[str] = None,
    ) -> List[mlflow.entities.Run]:
        """Search for runs with optional filtering and pagination."""
        kwargs: Dict[str, Any] = dict(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            run_view_type=run_view_type,
            max_results=max_results,
        )
        if order_by:
            kwargs["order_by"] = order_by
        return self.client.search_runs(**kwargs)

    def get_run(self, run_id: str) -> Optional[mlflow.entities.Run]:
        """Get a single run by ID."""
        try:
            return self.client.get_run(run_id)
        except Exception:
            return None

    def get_experiment(self, experiment_id: str) -> Optional[mlflow.entities.Experiment]:
        """Get an experiment by ID or name."""
        try:
            return self.client.get_experiment(experiment_id)
        except Exception:
            return None

    def get_metric_history(self, run_id: str, metric_key: str) -> List[mlflow.entities.Metric]:
        """Get the full metric history for a run."""
        return self.client.get_metric_history(run_id, metric_key)

    def get_run_params(self, run_id: str) -> Dict[str, str]:
        """Get all params for a run."""
        run = self.get_run(run_id)
        if run is None:
            return {}
        return {p.key: p.value for p in run.data.params}

    def get_run_metrics(self, run_id: str) -> Dict[str, List[mlflow.entities.Metric]]:
        """Get all metrics for a run, keyed by metric name."""
        run = self.get_run(run_id)
        if run is None:
            return {}
        return {m.key: m for m in run.data.metrics}

    def get_run_tags(self, run_id: str) -> Dict[str, str]:
        """Get all tags for a run."""
        run = self.get_run(run_id)
        if run is None:
            return {}
        return {t.key: t.value for t in run.data.tags}

    # --- ExperimentIQ-specific query methods ---

    def find_best_runs_by_metric(
        self,
        experiment_ids: List[str],
        metric_key: str,
        ascending: bool = True,
        max_results: int = 10,
        filter_string: str = "",
    ) -> List[mlflow.entities.Run]:
        """Find runs with the best (lowest or highest) value for a metric.
        """
        runs = self.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=[f"metrics.{metric_key} {'ASC' if ascending else 'DESC'}"],
        )
        return runs

    def find_runs_with_param_value(
        self,
        experiment_ids: List[str],
        param_key: str,
        param_value: str,
        max_results: int = 50,
    ) -> List[mlflow.entities.Run]:
        """Find runs that have a specific parameter value."""
        filter_string = f"params.{param_key} = '{param_value}'"
        return self.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
        )

    def get_runs_by_metric_range(
        self,
        experiment_ids: List[str],
        metric_key: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        max_results: int = 100,
    ) -> List[mlflow.entities.Run]:
        """Find runs with metric values in a range."""
        conditions = []
        if min_val is not None:
            conditions.append(f"metrics.{metric_key} >= {min_val}")
        if max_val is not None:
            conditions.append(f"metrics.{metric_key} <= {max_val}")

        filter_string = " AND ".join(conditions) if conditions else ""
        return self.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
        )
