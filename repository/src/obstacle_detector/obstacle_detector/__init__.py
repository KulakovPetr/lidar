"""Geometric obstacle contracts for a metro-train lidar. No neural network."""

from obstacle_detector.contracts import (
    MAX_IN_FLIGHT_FRAMES,
    MAX_QUEUE_DEPTH,
    Candidate,
    Corridor,
    Diagnostics,
    Frame,
    ObservationCoverage,
    ObstacleObservation,
    PathHypothesis,
    PathQuality,
    RouteSelection,
    unevaluated_diagnostics,
)

__all__ = [
    "MAX_IN_FLIGHT_FRAMES",
    "MAX_QUEUE_DEPTH",
    "Candidate",
    "Corridor",
    "Diagnostics",
    "Frame",
    "ObservationCoverage",
    "ObstacleObservation",
    "PathHypothesis",
    "PathQuality",
    "RouteSelection",
    "unevaluated_diagnostics",
]
