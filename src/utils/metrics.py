from typing import Dict, List, Any
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from collections import defaultdict

class MetricsAggregator:
    """Aggregates and analyzes metrics from LLM operations"""

    def __init__(self):
        self.metrics_history = defaultdict(list)

    def add_metrics(self, metrics: Dict[str, Any], timestamp: datetime = None) -> None:
        """Add metrics with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append((timestamp, value))

    def get_stats(self, metric_name: str, window: timedelta = None) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        values = self._get_values(metric_name, window)

        if not values:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "stddev": 0
            }

        try:
            return {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
                "stddev": stdev(values) if len(values) > 1 else 0
            }
        except Exception as e:
            print(f"Error calculating stats for {metric_name}: {e}")
            return {}

    def get_trends(self, metric_name: str, interval: timedelta = timedelta(hours=1)) -> List[Dict]:
        """Get metric trends over time intervals"""
        if metric_name not in self.metrics_history:
            return []

        # Group by interval
        intervals = defaultdict(list)
        for timestamp, value in self.metrics_history[metric_name]:
            interval_start = timestamp.replace(minute=0, second=0, microsecond=0)
            intervals[interval_start].append(value)

        # Calculate stats for each interval
        return [
            {
                "interval_start": start,
                "mean": mean(values),
                "count": len(values)
            }
            for start, values in sorted(intervals.items())
        ]

    def _get_values(self, metric_name: str, window: timedelta = None) -> List[float]:
        """Get values for a metric within time window"""
        if metric_name not in self.metrics_history:
            return []

        if window is None:
            return [value for _, value in self.metrics_history[metric_name]]

        cutoff = datetime.now() - window
        return [
            value for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff
        ]
