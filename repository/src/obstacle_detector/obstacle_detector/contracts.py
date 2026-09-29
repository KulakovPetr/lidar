"""Data contracts for the tunnel obstacle pipeline.

Four judgments stay separate: obstacle observation, path quality, route,
and observation coverage. A geometric score is not a probability unless a
calibration id is supplied. The tunnel center is not a valid route.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_QUEUE_DEPTH = 2
MAX_IN_FLIGHT_FRAMES = 2

ALLOWED_PATH_ORIGINS = frozenset({"rails", "ego_motion", "manual_prior", "unknown"})
FORBIDDEN_PATH_ORIGINS = frozenset(
    {
        "tunnel_center",
        "tunnel_centre",
        "tunnel_axis",
        "centerline",
        "centreline",
    }
)


def check_queue_depth(depth: int) -> int:
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise TypeError("queue depth must be an int")
    if depth < 1 or depth > MAX_QUEUE_DEPTH:
        raise ValueError(
            f"queue depth {depth} is outside 1..{MAX_QUEUE_DEPTH}"
        )
    return depth


@dataclass(frozen=True)
class Frame:
    """One PointCloud2 sweep. ring, per-point time, and IMU may be absent."""

    sequence: int
    frame_id: str
    stamp_ns: int | None
    point_count: int
    source_bag: str
    source_topic: str
    has_intensity: bool
    has_ring: bool = False
    has_point_time: bool = False
    has_imu: bool = False

    def __post_init__(self) -> None:
        if self.point_count < 0:
            raise ValueError("point_count must be >= 0")
        if not self.source_topic:
            raise ValueError("source_topic is required")


@dataclass(frozen=True)
class PathHypothesis:
    """Own-track hypothesis. Origin `tunnel_center` is rejected."""

    hypothesis_id: str
    frame_sequence: int
    origin: str
    polyline_xyz: tuple[tuple[float, float, float], ...] = ()
    quality: float | None = None
    supported_length_m: float | None = None

    def __post_init__(self) -> None:
        if self.origin in FORBIDDEN_PATH_ORIGINS or self.origin not in ALLOWED_PATH_ORIGINS:
            raise ValueError(
                f"path origin {self.origin!r} is not allowed; "
                "the tunnel center is not the train path"
            )
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("path quality must be in [0, 1] or None")
        if self.supported_length_m is not None and self.supported_length_m < 0.0:
            raise ValueError("supported_length_m must be measured and >= 0, or None")


@dataclass(frozen=True)
class Corridor:
    """Clearance relative to a path hypothesis, not relative to the tunnel axis.

    `s_far_m` is the far edge of *this* observation. There is no default horizon.
    Gauge width and height stay None until a measured or specified gauge exists.
    """

    path_id: str
    s_near_m: float | None = None
    s_far_m: float | None = None
    half_width_m: float | None = None
    height_above_rail_m: float | None = None

    def __post_init__(self) -> None:
        if (
            self.s_near_m is not None
            and self.s_far_m is not None
            and self.s_far_m < self.s_near_m
        ):
            raise ValueError("corridor s_far_m is closer than s_near_m")


@dataclass(frozen=True)
class Candidate:
    """One geometric observation. Score is uncalibrated unless calibration_id is set."""

    candidate_id: str
    frame_sequence: int
    point_count: int
    path_id: str | None = None
    centroid_xyz: tuple[float, float, float] | None = None
    distance_along_path_m: float | None = None
    extent_m: tuple[float, float, float] | None = None
    geometric_score: float | None = None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if self.point_count < 0:
            raise ValueError("point_count must be >= 0")
        if self.geometric_score is not None and self.geometric_score < 0.0:
            raise ValueError("geometric_score must be >= 0 or None")
        if self.calibration_id is not None and not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty when set")

    @property
    def score_is_probability(self) -> bool:
        return self.calibration_id is not None and self.geometric_score is not None

    @property
    def score_label(self) -> str:
        if self.geometric_score is None:
            return "absent"
        if self.score_is_probability:
            return "calibrated_probability"
        return "uncalibrated_geometric_score"


@dataclass(frozen=True)
class ObstacleObservation:
    candidates: tuple[Candidate, ...] = ()
    nearest_distance_m: float | None = None
    reported: bool | None = None


@dataclass(frozen=True)
class PathQuality:
    selected_quality: float | None = None
    supported_length_m: float | None = None
    reason: str = "not_estimated"


@dataclass(frozen=True)
class RouteSelection:
    path_id: str | None = None
    origin: str | None = None

    def __post_init__(self) -> None:
        if self.origin is None:
            return
        if self.origin in FORBIDDEN_PATH_ORIGINS or self.origin not in ALLOWED_PATH_ORIGINS:
            raise ValueError(
                f"route origin {self.origin!r} is not allowed; "
                "the tunnel center is not the train path"
            )


@dataclass(frozen=True)
class ObservationCoverage:
    """What this frame actually returned. No default rail-visibility distance."""

    max_return_range_m: float | None = None
    path_supported_range_m: float | None = None
    missing_optional_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnostics:
    obstacle: ObstacleObservation
    path_quality: PathQuality
    route: RouteSelection
    coverage: ObservationCoverage
    queue_depth: int
    dropped_frames: int
    processing_ms: float | None = None

    def __post_init__(self) -> None:
        check_queue_depth(self.queue_depth)
        if self.dropped_frames < 0:
            raise ValueError("dropped_frames must be >= 0")


def unevaluated_diagnostics(queue_depth: int = 1, dropped_frames: int = 0) -> Diagnostics:
    """Explicit non-result. Does not claim the track is clear."""

    return Diagnostics(
        obstacle=ObstacleObservation(),
        path_quality=PathQuality(),
        route=RouteSelection(),
        coverage=ObservationCoverage(),
        queue_depth=queue_depth,
        dropped_frames=dropped_frames,
        processing_ms=None,
    )
