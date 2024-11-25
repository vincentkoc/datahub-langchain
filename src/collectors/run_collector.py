from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..base import LLMRun, LLMPlatformConnector

class RunCollector:
    """Collects run information from various platforms"""

    def __init__(self, connectors: List[LLMPlatformConnector]):
        self.connectors = connectors

    def collect_runs(self, start_time: datetime = None, end_time: datetime = None,
                    limit: int = 1000) -> List[LLMRun]:
        """Collect runs from all configured platforms"""
        runs = []
        for connector in self.connectors:
            try:
                platform_runs = connector.get_runs(
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                runs.extend(platform_runs)
            except Exception as e:
                print(f"Error collecting runs from {connector.__class__.__name__}: {e}")
        return runs

    def get_run_stats(self, time_window: timedelta) -> Dict[str, Any]:
        """Get statistics about collected runs"""
        end_time = datetime.now()
        start_time = end_time - time_window
        runs = self.collect_runs(start_time=start_time, end_time=end_time)

        return {
            "total_runs": len(runs),
            "success_rate": self._compute_success_rate(runs),
            "average_latency": self._compute_average_latency(runs),
            "total_cost": self._compute_total_cost(runs),
            "error_distribution": self._compute_error_distribution(runs)
        }

    def _compute_success_rate(self, runs: List[LLMRun]) -> float:
        if not runs:
            return 0.0
        successful = sum(1 for run in runs if run.metrics.get("status") == "success")
        return successful / len(runs)

    def _compute_average_latency(self, runs: List[LLMRun]) -> float:
        latencies = [run.metrics.get("latency", 0) for run in runs if run.metrics.get("latency")]
        return sum(latencies) / len(latencies) if latencies else 0

    def _compute_total_cost(self, runs: List[LLMRun]) -> float:
        return sum(run.metrics.get("cost", 0) for run in runs)

    def _compute_error_distribution(self, runs: List[LLMRun]) -> Dict[str, int]:
        errors = {}
        for run in runs:
            if error := run.metrics.get("error"):
                error_type = error.get("type", "unknown")
                errors[error_type] = errors.get(error_type, 0) + 1
        return errors
