"""Active analysis and explicit historical CIFAR-10 compatibility exports."""

from orbit_cifar.analysis.active import aggregate_active, analyze_active
from orbit_cifar.analysis.legacy_aggregate import aggregate_outputs

__all__ = ["aggregate_active", "analyze_active", "aggregate_outputs"]
