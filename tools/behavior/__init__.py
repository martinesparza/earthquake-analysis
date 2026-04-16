"""
Behavior Analysis Toolkit
Phase 1: Session metadata and data organization
Phase 2: Cross-session comparisons
"""

from .session_metadata import SessionMetadata, SessionRegistry, load_session_registry
from .behavior_dataset import BehaviorDataset
from .time_series_plotter import TimeSeriesPlotter
from .session_comparator import SessionComparator

__all__ = [
    "SessionMetadata",
    "SessionRegistry",
    "load_session_registry",
    "BehaviorDataset",
    "TimeSeriesPlotter",
    "SessionComparator",
]
