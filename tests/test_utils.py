import pytest
from datetime import datetime, timedelta
from src.utils.metrics import MetricsAggregator
from src.utils.lineage import LineageTracker
from src.utils.model_utils import get_capabilities_from_model, get_provider_from_model

def test_metrics_aggregator():
    aggregator = MetricsAggregator()

    # Add test metrics
    metrics = {
        "latency": 1.0,
        "tokens": 100,
        "cost": 0.01
    }
    aggregator.add_metrics(metrics)

    # Test statistics
    stats = aggregator.get_stats("latency")
    assert stats["count"] == 1
    assert stats["mean"] == 1.0

    # Test trends
    trends = aggregator.get_trends("latency")
    assert len(trends) == 1
    assert "mean" in trends[0]

def test_lineage_tracker():
    tracker = LineageTracker()

    # Add test edges
    tracker.add_edge("model1", "run1", "USES")
    tracker.add_edge("run1", "chain1", "PART_OF")

    # Test relationships
    upstream = tracker.get_upstream("run1")
    assert len(upstream) == 1
    assert upstream[0].source_id == "model1"

    downstream = tracker.get_downstream("run1")
    assert len(downstream) == 1
    assert downstream[0].target_id == "chain1"

    # Test full graph
    graph = tracker.get_lineage_graph()
    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2

def test_model_utils():
    # Test capability detection
    gpt4_capabilities = get_capabilities_from_model("gpt-4")
    assert "text-generation" in gpt4_capabilities
    assert "function-calling" in gpt4_capabilities

    # Test provider detection
    assert get_provider_from_model("gpt-4") == "OpenAI"
    assert get_provider_from_model("claude-2") == "Anthropic"
    assert get_provider_from_model("unknown-model") == "Unknown"
