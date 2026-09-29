"""ROS 2 adapter. The subscription callback does not run the detector.

Stream mode keeps one queued cloud and counts the ones it replaces.
Sequential mode does not replace a cloud. It waits for /frame_token and
processes that pair. The token sequence is the message identity; the stamp
is only stored.

/detection carries a short card for every finished frame. The full record is
appended to frames.jsonl. The picture is a separate output: one waiting
cloud, published at visualization_rate_hz. A display cloud is a stride sample
plus every accepted point. That sample is not the detector input.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from obstacle_detector.bag_io import decode_message, decode_recorded
from obstacle_detector.display_cloud import highlight_rows, sample_display
from obstacle_detector.display_state import annotate_display, processed_marker_text
from obstacle_detector.frame_budget import CloudBuffers, WorkerPool, call_bounded, ensure_worker
from obstacle_detector.frame_pipeline import algorithm_version, config_dir, load_runtime, variant_flags
from obstacle_detector.offline_run import execute_grouped_frame, load_mount_file, process_stream_job
from obstacle_detector.repeat_groups import group_exact_xyz, valid_mask
from obstacle_detector.straight_corridor import load_profile
from obstacle_detector.working import config_version


def _viz_prepare_job(job: dict) -> dict:
    """Decode and thin one cloud in a child process so the node keeps the GIL."""
    from multiprocessing import shared_memory

    started = time.perf_counter()
    if job.get("xyz") is not None:
        xyz = np.asarray(job["xyz"], dtype=np.float32)
    else:
        shm = shared_memory.SharedMemory(name=job["name"])
        try:
            raw = np.ndarray((int(job["nbytes"]),), dtype=np.uint8, buffer=shm.buf).copy()
        finally:
            shm.close()
        decoded = decode_recorded(raw, job["spec"])
        xyz = np.column_stack((decoded["x"], decoded["y"], decoded["z"])).astype(np.float32, copy=False)
    shown, labels = sample_display(xyz, np.asarray(job["highlight"], dtype=np.int64), int(job["limit"]))
    return {
        "ok": True,
        "generation": job["generation"],
        "sequence": job["sequence"],
        "shown": np.asarray(shown, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "prepare_s": time.perf_counter() - started,
        "frame_id": job["frame_id"],
        "stamp_ns": int(job["stamp_ns"]),
        "view": job["view"],
        "ready_mono": job["ready_mono"],
    }


def _viz_serve(inbox, outbox) -> None:
    from obstacle_detector.frame_budget import limit_library_threads

    limit_library_threads()
    while True:
        job = inbox.get()
        if job is None:
            return
        try:
            outbox.put(_viz_prepare_job(job))
        except Exception as exc:  # noqa: BLE001
            outbox.put({"ok": False, "generation": job.get("generation"), "error": f"{type(exc).__name__}: {exc}"})


def _visualization_rate_hz(value) -> float:
    """Maximum rate of a new picture. Zero, negative and non-numeric values are rejected."""
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("visualization_rate_hz должен быть числом") from exc
    if rate != rate or rate <= 0.0 or rate > 60.0:
        raise RuntimeError("visualization_rate_hz должен быть положительным и не больше 60")
    return rate


def journal_key(record: dict) -> tuple[int, int]:
    """One result identity: session and callback sequence. A header stamp is not the key."""
    session = int(record.get("session_id") or 0)
    sequence = record.get("sequence")
    if sequence is None:
        sequence = record.get("message_index")
    return session, int(sequence)


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"не сериализуется: {type(value).__name__}")


def _clear_marker(header) -> Marker:
    marker = Marker()
    marker.header.stamp = header.stamp
    marker.header.frame_id = header.frame_id
    marker.action = Marker.DELETEALL
    return marker


def _text_marker(header, text: str, z: float) -> Marker:
    marker = Marker()
    marker.header.stamp = header.stamp
    marker.header.frame_id = header.frame_id
    marker.ns = "decision"
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.z = z
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.35
    marker.color.r = marker.color.g = marker.color.b = 1.0
    marker.color.a = 1.0
    marker.text = text
    return marker


def _markers(record: dict, header, text: str) -> MarkerArray:
    array = MarkerArray()
    array.markers.append(_clear_marker(header))
    array.markers.append(_text_marker(header, text, 2.0))
    if record.get("display_role") == "processing_error" or record.get("status") != "ok":
        return array
    accepted = {item.get("candidate_id") for item in record.get("conditional_intrusions") or []}
    for ordinal, item in enumerate(record.get("candidates") or []):
        position = item.get("sensor_position_m") or {}
        if "x" not in position:
            continue
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = header.frame_id
        marker.ns = "conditional_intrusion" if item.get("candidate_id") in accepted else "candidate_only"
        marker.id = ordinal + 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position["x"])
        marker.pose.position.y = float(position["y"])
        marker.pose.position.z = float(position["z"])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.45 if item.get("candidate_id") in accepted else 0.25
        if item.get("candidate_id") in accepted:
            marker.color.r, marker.color.g, marker.color.b = 0.9, 0.15, 0.1
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.6, 0.6, 0.6
        marker.color.a = 0.9
        array.markers.append(marker)
    return array


def _display_message(header, xyz: np.ndarray, labels: np.ndarray, sequence: int) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = header.stamp
    msg.header.frame_id = f"{header.frame_id}#{int(sequence)}"
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * msg.width
    msg.is_dense = True
    packed = np.empty(
        msg.width,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<f4")]),
    )
    if msg.width:
        packed["x"] = xyz[:, 0]
        packed["y"] = xyz[:, 1]
        packed["z"] = xyz[:, 2]
        packed["label"] = labels
    msg.data = packed.tobytes()
    return msg


def _range_bound(record: dict, reducer):
    if record.get("candidates") is None:
        return None
    values = [
        float(item["range_from_lidar_m"])
        for item in record.get("candidates") or []
        if item.get("range_from_lidar_m") is not None
    ]
    if not values:
        return None
    return round(float(reducer(values)), 4)


def display_slot_accepts(current: dict | None, session: int, reception: int) -> bool:
    """A picture slot keeps only a newer result. A late reply does not roll it back."""
    if current is None:
        return True
    if int(session) != int(current["session"]):
        return int(session) > int(current["session"])
    return int(reception) > int(current["reception"])


def picture_update(slot: dict | None, status: str, session: int, reception: int):
    """Timeout and error do not replace the cloud. They only note the latest result."""
    if status != "ok":
        return slot, {"status": status, "session": int(session), "reception": int(reception)}
    if not display_slot_accepts(slot, session, reception):
        return slot, None
    return {"session": int(session), "reception": int(reception), "status": "ok"}, None


def _viz_view(record: dict) -> dict:
    """Fields the picture needs. Source-index lists stay in the journal record."""
    accepted = [{"candidate_id": item.get("candidate_id")} for item in record.get("conditional_intrusions") or []]
    candidates = [
        {"candidate_id": item.get("candidate_id"), "sensor_position_m": item.get("sensor_position_m")}
        for item in record.get("candidates") or []
    ]
    return {
        "status": record.get("status"),
        "decision": record.get("decision"),
        "display_role": record.get("display_role"),
        "display_label": record.get("display_label"),
        "session_id": record.get("session_id"),
        "header_stamp_ns": record.get("header_stamp_ns"),
        "header_frame_id": record.get("header_frame_id"),
        "processing_id": record.get("processing_id"),
        "result_age_receptions": record.get("result_age_receptions"),
        "sequence": record.get("sequence"),
        "conditional_intrusions": accepted,
        "candidates": candidates,
    }


def _picture_text(view: dict, age_s: float, latest: dict | None) -> str:
    text = processed_marker_text(view)
    lines = [text, f"возраст рисунка {age_s:.1f} с"]
    if latest and (latest.get("sequence") != view.get("sequence") or latest.get("status") != "ok"):
        lines.append(
            f"последний результат {latest.get('sequence')} {latest.get('status')}; "
            "этот рисунок не является ответом на него"
        )
    return "\n".join(lines)


def _compact(record: dict, snapshot: dict) -> dict:
    intrusions = record.get("conditional_intrusions") or []
    first = intrusions[0] if intrusions else {}
    profile = record.get("profile") or {}
    timing = record.get("timing_s") or {}
    kept = {
        "decode": timing.get("decode"),
        "group": timing.get("group"),
        "queue_wait_s": timing.get("queue_wait_s"),
        "geometry": timing.get("geometry"),
        "surface": timing.get("surface"),
        "submit_to_collect_s": timing.get("submit_to_collect_s"),
        "compact_publish_s": timing.get("compact_publish_s"),
        "worker_and_publish_prep": timing.get("worker_and_publish_prep"),
        "frame_excluding_bag_read": timing.get("frame_excluding_bag_read"),
        "detector": timing.get("detector"),
        "connectivity": timing.get("connectivity"),
        "corridor": timing.get("corridor"),
        "filters": timing.get("filters"),
        "display_prepare_s": timing.get("display_prepare_s"),
        "serialize_s": timing.get("serialize_s"),
        "publish_s": timing.get("publish_s"),
    }
    return {
        "run_id": record.get("run_id"),
        "sequence": record.get("sequence"),
        "frame_index": record.get("frame_index"),
        "header_stamp_ns": record.get("header_stamp_ns"),
        "header_frame_id": record.get("header_frame_id"),
        "bag_time_ns": record.get("bag_time_ns"),
        "status": record.get("status"),
        "decision": record.get("decision"),
        "display_role": record.get("display_role"),
        "display_label": record.get("display_label"),
        "range_from_lidar_m": first.get("range_from_lidar_m"),
        "evidence": first.get("evidence"),
        "corridor_segment": first.get("corridor_segment"),
        "intrusion_count": len(intrusions),
        "highlight_count": int(record.get("highlight_count") or 0),
        "result_elapsed_s": record.get("result_elapsed_s"),
        "latency_from_arrival_s": record.get("latency_from_arrival_s"),
        "latency_to_compact_s": record.get("latency_to_compact_s"),
        "newer_input_exists": bool(record.get("newer_input_exists")),
        "result_is_answer_to_latest_input": bool(record.get("result_is_answer_to_latest_input")),
        "job_id": record.get("job_id"),
        "session_id": record.get("session_id"),
        "algorithm_version": record.get("algorithm_version"),
        "config_version": record.get("config_version"),
        "candidate_count": None if record.get("candidates") is None else len(record.get("candidates") or []),
        "range_min_m": _range_bound(record, min),
        "range_max_m": _range_bound(record, max),
        "result_matches_processed_cloud": False,
        "compact_result_is_not_the_picture": True,
        "timeout_is_not_absence_of_obstacle": record.get("status") == "processing_timeout",
        "playback_mode": record.get("playback_mode"),
        "processing_id": record.get("processing_id"),
        "timing_s": kept,
        "profile": {
            "source": profile.get("corridor_source") or record.get("corridor_source"),
            "s_max_m": profile.get("s_max_m"),
            "far_assumption_from_s_m": profile.get("far_assumption_from_s_m"),
            "search_horizon_is_not_a_confirmed_path": profile.get("search_horizon_is_not_a_confirmed_path"),
            "estimated_trajectory": profile.get("estimated_trajectory"),
            "position_uncertainty": profile.get("position_uncertainty"),
            "prior": profile.get("prior"),
            "prior_is_assumption": profile.get("prior_is_assumption"),
        },
        "counts": {
            "received": snapshot.get("received"),
            "completed_ok": snapshot.get("completed_ok"),
            "dropped": snapshot.get("dropped"),
            "processing_error": snapshot.get("processing_error"),
            "processing_timeout": snapshot.get("processing_timeout"),
            "queued": snapshot.get("queued"),
            "in_flight": snapshot.get("in_flight"),
        },
        "full_record_is_on_disk": True,
        "display_cloud_is_not_the_detector_input": True,
    }


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_detector")
        self.declare_parameter("input_topic", "/lidar_points")
        self.declare_parameter("corridor_source", "inferred")
        self.declare_parameter("profile", "")
        self.declare_parameter("mount", "")
        self.declare_parameter("frame_budget_s", 30.0)
        self.declare_parameter("run_detector", True)
        self.declare_parameter("budget_is_artificial_timeout", False)
        self.declare_parameter("playback_mode", "stream")
        self.declare_parameter("output_dir", "/output")
        self.declare_parameter("compute_backend", "cpu")
        self.declare_parameter("queue_policy", "keep_latest")
        self.declare_parameter("subscription_depth", 10)
        self.declare_parameter("publish_visualization", True)
        self.declare_parameter("visualization_enabled", True)
        self.declare_parameter("visualization_rate_hz", "2.0")
        self.declare_parameter("display_background_limit", 3500)
        self.declare_parameter("workers", 2)
        topic = str(self.get_parameter("input_topic").value)
        self._source = str(self.get_parameter("corridor_source").value)
        self._budget = float(self.get_parameter("frame_budget_s").value)
        self._artificial_budget = _flag(self.get_parameter("budget_is_artificial_timeout").value)
        self._run_detector = _flag(self.get_parameter("run_detector").value)
        mode = str(self.get_parameter("playback_mode").value or "stream").strip().lower()
        if mode not in ("stream", "sequential"):
            raise RuntimeError("playback_mode должен быть stream или sequential")
        self._mode = mode
        self._backend = str(self.get_parameter("compute_backend").value or "cpu").strip().lower()
        if self._backend not in ("cpu", "gpu"):
            raise RuntimeError("compute_backend должен быть cpu или gpu")
        self._backend_requested = self._backend
        if self._backend == "gpu":
            self.get_logger().error(
                "compute_backend=gpu недоступен: Linux CUDA-эксперимент не ускорил кадр и не подключён. "
                "Детектор остаётся на cpu и не подменяет GPU молча."
            )
            self._backend = "cpu"
        self._queue_policy = str(self.get_parameter("queue_policy").value or "keep_latest").strip().lower()
        if self._mode == "stream" and self._queue_policy != "keep_latest":
            raise RuntimeError("потоковый режим держит одно последнее облако: queue_policy=keep_latest")
        depth = int(self.get_parameter("subscription_depth").value)
        if depth < 1 or depth > 10:
            self.get_logger().warn(f"subscription_depth {depth} заменён на 10. Бесконечная очередь не используется.")
            depth = 10
        self._subscription_depth = depth
        self._publish_visualization = _flag(self.get_parameter("publish_visualization").value)
        self._viz_enabled = self._publish_visualization and _flag(self.get_parameter("visualization_enabled").value)
        self._viz_rate = _visualization_rate_hz(self.get_parameter("visualization_rate_hz").value)
        self._display_limit = int(self.get_parameter("display_background_limit").value)
        self._workers = int(self.get_parameter("workers").value)
        if self._workers not in (1, 2, 4, 6):
            raise RuntimeError("workers должен быть 1, 2, 4 или 6")
        os.environ["OBSTACLE_COMPUTE_BACKEND"] = self._backend
        mount_path = str(self.get_parameter("mount").value) or str(config_dir() / "mount_unconfirmed.yaml")
        self._corridor, self._mount = load_mount_file(Path(mount_path))
        _packaged, self._detector, batch, _unused_mount = load_runtime()
        self._flags = variant_flags(batch, batch["default_variant"])
        profile = str(self.get_parameter("profile").value)
        if self._source == "configured_straight" and not profile:
            profile = str(config_dir() / "configured_straight.yaml")
        self._straight = load_profile(Path(profile)) if self._source == "configured_straight" else None
        if self._source == "configured_straight" and self._straight is None:
            raise RuntimeError("configured_straight требует profile")
        self._output_dir = Path(str(self.get_parameter("output_dir").value or "/output"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._claim_output_dir()
        self._jsonl_path = self._output_dir / "frames.jsonl"
        self._jsonl = self._jsonl_path.open("x", encoding="utf-8")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._subscription_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._counts = {
            "received": 0,
            "queued": 0,
            "in_flight": 0,
            "completed_ok": 0,
            "dropped": 0,
            "processing_error": 0,
            "processing_timeout": 0,
            "observed_only": 0,
            "awaiting_pair": 0,
        }
        self._slot = None
        self._arrived = {}
        self._seq_queue = []
        self._token_q = []
        self._cloud_q = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._session_id = 1
        self._last_stamp_ns = None
        self._reception = 0
        self.create_subscription(PointCloud2, topic, self._on_cloud, qos)
        self.create_subscription(String, "/frame_token", self._on_token, qos)
        self._result = self.create_publisher(String, "detection", qos)
        self._markers = self.create_publisher(MarkerArray, "markers", qos)
        self._cloud = self.create_publisher(PointCloud2, "processed_cloud", qos)
        self._display = self.create_publisher(PointCloud2, "display_cloud", qos)
        self._rviz_cloud = self.create_publisher(PointCloud2, "rviz_cloud", qos)
        self._previous_markers = self.create_publisher(MarkerArray, "previous_markers", qos)
        self._previous_cloud = self.create_publisher(PointCloud2, "previous_cloud", qos)
        self._diag = self.create_publisher(String, "diagnostics", qos)
        self._processing_id = 0
        self._last_success = None
        self._displayed_session = None
        self._shown_reception = {}
        self._jobs = {}
        self._deadlines = {}
        self._next_job = 0
        self._pool = None
        self._buffers = None
        if self._mode == "stream" and self._run_detector:
            self._pool = WorkerPool(self._workers)
            picture_slots = 1 if self._viz_enabled else 0
            self._buffers = CloudBuffers(self._workers + picture_slots, 48 * 1024 * 1024)
        else:
            ensure_worker()
        self._viz_lock = threading.Lock()
        self._viz_slot = None
        self._viz_waiting = False
        self._viz_reset_session = None
        self._latest_result = None
        self._shown = None
        self._rate_marks = {
            "input": deque(),
            "completed": deque(),
            "compact": deque(),
            "viz_prepared": deque(),
            "viz_published": deque(),
        }
        self._viz_counts = {"prepared": 0, "published": 0, "replaced": 0, "skipped_unwatched": 0}
        self._journal_counts = {"written": 0, "waited": 0, "errors": 0, "duplicates_rejected": 0, "closed": 0}
        self._journal_keys: set[tuple[int, int]] = set()
        self._journal_q = queue.Queue(maxsize=4)
        self._journal_file_lock = threading.Lock()
        self._timing_handle = (self._output_dir / "publish_timing.jsonl").open("x", encoding="utf-8")
        self._viz_timing = (self._output_dir / "viz_timing.jsonl").open("x", encoding="utf-8")
        self._journal_thread = threading.Thread(target=self._journal_loop, name="detector-journal", daemon=True)
        self._journal_thread.start()
        self._viz_generation = 0
        self._viz_busy = False
        self._viz_inflight = None
        self._viz_in = None
        self._viz_out = None
        self._viz_process = None
        if self._viz_enabled:
            context = mp.get_context("spawn")
            self._viz_in = context.Queue(maxsize=1)
            self._viz_out = context.Queue(maxsize=1)
            self._viz_process = context.Process(target=_viz_serve, args=(self._viz_in, self._viz_out), daemon=True)
            self._viz_process.start()
        self._viz_thread = threading.Thread(target=self._viz_loop, name="detector-visualization", daemon=True)
        self._viz_thread.start()
        self._thread = threading.Thread(target=self._supervise, name="detector-supervisor", daemon=True)
        self._thread.start()
        self.create_timer(0.5, self._publish_diag)
        self._identity = self._run_identity()
        self.get_logger().info(
            "subscription_ready "
            f"input_topic={topic} corridor_source={self._source} "
            f"profile={self._identity['profile_name']} "
            f"profile_limits={self._identity['profile_limits']} "
            f"profile_is_not_a_confirmed_path=true "
            f"workers={self._workers} compute_backend={self._backend} "
            f"algorithm_version={self._identity['algorithm_version']} "
            f"config_version={self._identity['config_version']} "
            f"playback_mode={self._mode} queue_policy={self._queue_policy} "
            f"subscription_depth={self._subscription_depth} "
            f"requested_backend={self._backend_requested} "
            f"rmw={os.environ.get('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')} "
            f"domain={os.environ.get('ROS_DOMAIN_ID', '')} "
            f"run_detector={self._run_detector} frame_budget_s={self._budget} "
            f"visualization_enabled={self._viz_enabled} visualization_rate_hz={self._viz_rate} "
            "node_does_not_open_a_bag=true"
        )

    def _run_identity(self) -> dict:
        profile = self._straight or {}
        limits = "none"
        if profile:
            limits = (
                f"s={profile.get('s_min_m')}..{profile.get('s_max_m')}m "
                f"width={profile.get('width_m')}m height={profile.get('height_m')}m "
                f"center_l={profile.get('center_l_m')}m "
                f"assumption_not_confirmed_path"
            )
        return {
            "corridor_source": self._source,
            "profile_name": profile.get("source_path_name") or "none",
            "profile_limits": limits,
            "profile_is_not_a_confirmed_path": True,
            "workers": self._workers,
            "compute_backend": self._backend,
            "algorithm_version": algorithm_version(),
            "config_version": config_version(self._detector, self._corridor, self._mount, self._flags, None),
        }

    def _snapshot(self) -> dict:
        payload = dict(self._counts)
        payload["playback_mode"] = self._mode
        payload["queue_depth"] = len(self._seq_queue) if self._mode == "sequential" else (1 if self._slot is not None else 0)
        payload["sequential_queue_does_not_drop"] = self._mode == "sequential"
        payload["unbounded_queue"] = False
        payload["stream_keeps_latest_only"] = self._mode == "stream"
        payload["processed_includes_errors"] = False
        payload["processed_field_is"] = "completed_ok"
        payload["not_a_processing_rate_hz"] = True
        accounted = (
            payload["completed_ok"]
            + payload["dropped"]
            + payload["processing_error"]
            + payload["processing_timeout"]
            + payload["queued"]
            + payload["in_flight"]
            + payload["observed_only"]
            + payload["awaiting_pair"]
        )
        payload["accounted"] = accounted
        payload["balance_ok"] = accounted == payload["received"]
        payload["session_id"] = self._session_id
        payload["budget_is_artificial_timeout"] = self._artificial_budget
        payload["visualization_enabled"] = self._viz_enabled
        payload["visualization_rate_hz"] = self._viz_rate
        payload["input_hz"] = self._rate_unlocked("input")
        payload["completed_hz"] = self._rate_unlocked("completed")
        payload["compact_hz"] = self._rate_unlocked("compact")
        payload["viz_prepared_hz"] = self._rate_unlocked("viz_prepared")
        payload["viz_published_hz"] = self._rate_unlocked("viz_published")
        payload["rates_are_separate"] = True
        shown = self._shown or {}
        latest = self._latest_result or {}
        payload["shown_age_s"] = shown.get("age_s")
        payload["shown_sequence"] = shown.get("sequence")
        payload["shown_status"] = shown.get("status")
        payload["shown_is_latest_result"] = bool(shown) and shown.get("sequence") == latest.get("sequence") and latest.get("status") == "ok"
        payload["viz_prepared_total"] = self._viz_counts["prepared"]
        payload["viz_published_total"] = self._viz_counts["published"]
        payload["viz_replaced_total"] = self._viz_counts["replaced"]
        payload["viz_skipped_unwatched_total"] = self._viz_counts["skipped_unwatched"]
        payload["journal_written"] = self._journal_counts["written"]
        payload["journal_waits"] = self._journal_counts["waited"]
        payload["journal_errors"] = self._journal_counts["errors"]
        payload["journal_duplicates_rejected"] = self._journal_counts["duplicates_rejected"]
        payload["journal_closed"] = self._journal_counts["closed"] == 1
        payload["journal_saved_means_flushed"] = True
        payload["visualization_rate_limit_hz"] = self._viz_rate
        payload["viz_new_hz"] = payload["viz_prepared_hz"]
        payload["redraw_is_not_a_new_frame"] = True
        payload["message_index_is_callback_ordinal"] = True
        payload["visual_queue_depth"] = 1 if self._viz_waiting else 0
        payload.update(getattr(self, "_identity", {}))
        return payload

    def _publish_diag(self) -> None:
        with self._lock:
            payload = self._snapshot()
        self._diag.publish(String(data=json.dumps(payload)))

    def _match_sequential(self) -> None:
        while self._token_q and self._cloud_q:
            self._seq_queue.append((self._cloud_q.pop(0), self._token_q.pop(0)))
        self._counts["queued"] = len(self._seq_queue)
        self._counts["awaiting_pair"] = len(self._cloud_q)

    def _on_token(self, msg: String) -> None:
        if self._mode != "sequential":
            return
        try:
            token = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._token_q.append(token)
            self._match_sequential()

    def _on_cloud(self, msg: PointCloud2) -> None:
        stamp = _stamp_ns(msg.header)
        with self._lock:
            self._counts["received"] += 1
            if self._last_stamp_ns is not None and stamp < self._last_stamp_ns:
                self._session_id += 1
                self._viz_reset_session = self._session_id
            self._last_stamp_ns = stamp
            self._mark_unlocked("input")
            self._reception += 1
            reception = self._reception
            self._arrived[reception] = time.perf_counter()
            session = self._session_id
            if not self._run_detector:
                self._counts["observed_only"] += 1
            elif self._mode == "sequential":
                self._cloud_q.append((msg, reception, session, stamp))
                self._match_sequential()
            else:
                if self._slot is not None:
                    self._counts["dropped"] += 1
                self._slot = (msg, reception, session, stamp, None)
                self._counts["queued"] = 1

    def _take(self):
        with self._lock:
            if self._mode == "sequential":
                if not self._seq_queue:
                    return None
                cloud_item, token = self._seq_queue.pop(0)
                msg, reception, session, stamp = cloud_item
                self._counts["queued"] = len(self._seq_queue)
                self._counts["in_flight"] = 1
                self._counts["awaiting_pair"] = len(self._cloud_q)
                return msg, reception, session, stamp, token
            item = self._slot
            self._slot = None
            if item is None:
                return None
            self._counts["queued"] = 0
            self._counts["in_flight"] = 1
            return item

    def _supervise(self) -> None:
        while not self._stop.is_set():
            if self._mode == "stream" and self._run_detector and self._pool is not None:
                self._collect_stream()
                if self._pool.has_free():
                    item = self._take()
                    if item is not None:
                        self._submit_stream(item)
                        continue
                time.sleep(0.01)
                continue
            item = self._take()
            if item is None:
                time.sleep(0.02)
                continue
            msg, reception, session, stamp, token = item
            if self._mode == "sequential" and isinstance(token, dict):
                sequence = int(token.get("sequence"))
            else:
                sequence = int(reception - 1)
                token = None
            published = {"ok": False, "counted": False}
            try:
                self._finish_frame(msg, reception, session, stamp, token, sequence, published)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"frame {sequence} failed: {type(exc).__name__}: {exc}")
                with self._lock:
                    self._counts["in_flight"] = 0
                    if not published["counted"]:
                        self._counts["processing_error"] += 1
                if not published["ok"]:
                    run_id = token.get("run_id") if isinstance(token, dict) else None
                    card = {
                        "run_id": run_id,
                        "sequence": sequence,
                        "frame_index": sequence,
                        "header_stamp_ns": stamp,
                        "header_frame_id": str(msg.header.frame_id),
                        "status": "processing_error",
                        "decision": "processing_error",
                        "highlight_count": 0,
                        "playback_mode": self._mode,
                        "result_is_answer_to_latest_input": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "full_record_is_on_disk": False,
                    }
                    try:
                        self._result.publish(String(data=json.dumps(card, ensure_ascii=False)))
                    except Exception:
                        pass
            continue

    def _finish_frame(self, msg, reception, session, stamp, token, sequence, published) -> None:
        started = time.perf_counter()
        arrived = self._arrived.pop(reception, started)
        record, xyz = self._one_frame(msg, sequence)
        elapsed = time.perf_counter() - started
        record["reception_index"] = reception
        record["session_id"] = session
        record["header_stamp_ns"] = stamp
        record["header_frame_id"] = str(msg.header.frame_id)
        record["frame_index"] = sequence
        record["message_index"] = sequence
        record["callback_sequence"] = sequence
        record["message_index_is_callback_ordinal"] = True
        record["message_index_is_not_a_bag_index"] = True
        record["sequence"] = sequence
        record["playback_mode"] = self._mode
        record["result_elapsed_s"] = elapsed
        record.setdefault("timing_s", {})
        record["timing_s"]["queue_wait_s"] = started - arrived
        record["_arrival_mono"] = arrived
        record["budget_is_artificial_timeout"] = self._artificial_budget
        record["internal_format_is_not_a_contest_api"] = True
        if isinstance(token, dict):
            record["run_id"] = token.get("run_id")
            record["bag_time_ns"] = token.get("bag_time_ns")
            record["player"] = token.get("player")
        else:
            record["run_id"] = None
            record["bag_time_ns"] = None
            record["player"] = "ros2_bag_play" if self._mode == "stream" else None
        with self._lock:
            if self._mode == "sequential":
                superseded = False
            else:
                superseded = self._slot is not None or reception != self._reception
            self._counts["in_flight"] = 0
            if record.get("status") == "processing_timeout":
                self._counts["processing_timeout"] += 1
            elif record.get("status") == "processing_error":
                self._counts["processing_error"] += 1
            else:
                self._counts["completed_ok"] += 1
            published["counted"] = True
            self._mark_unlocked("completed")
            snapshot = self._snapshot()
        with self._lock:
            self._processing_id += 1
            processing_id = self._processing_id
            received_now = self._counts["received"]
            displayed_session = self._displayed_session
        annotate_display(
            record,
            processing_id=processing_id,
            session_id=session,
            stamp_ns=stamp,
            frame_id=msg.header.frame_id,
            superseded=superseded,
            received_now=received_now,
            reception_index=reception,
        )
        record["result_is_current"] = False
        record["result_is_answer_to_latest_input"] = not superseded
        if displayed_session is not None and session != displayed_session:
            self._clear_previous(msg.header)
        self._displayed_session = session
        self._emit(msg, record, xyz, snapshot, published)

    def _rate_unlocked(self, key: str) -> float:
        marks = self._rate_marks[key]
        if len(marks) < 2:
            return 0.0
        span = marks[-1] - marks[0]
        if span <= 0.0:
            return 0.0
        return round((len(marks) - 1) / span, 3)

    def _mark_unlocked(self, key: str, counter: str | None = None) -> None:
        now = time.perf_counter()
        marks = self._rate_marks[key]
        marks.append(now)
        cutoff = now - 5.0
        while marks and marks[0] < cutoff:
            marks.popleft()
        if counter:
            self._viz_counts[counter] = int(self._viz_counts.get(counter, 0)) + 1

    def _mark(self, key: str, counter: str | None = None) -> None:
        with self._lock:
            self._mark_unlocked(key, counter)

    def _emit(self, msg: PointCloud2, record: dict, xyz: np.ndarray, snapshot: dict, published: dict) -> None:
        arrived = record.pop("_arrival_mono", None)
        self._publish_compact(record, snapshot, published, arrived)
        self._offer_picture(record, xyz=xyz, header=msg.header)
        self._enqueue_journal(record)

    def _publish_compact(self, record: dict, snapshot: dict, published: dict, arrived) -> None:
        highlight = highlight_rows(record)
        record["highlight_count"] = int(highlight.size)
        record["_highlight"] = highlight
        card = _compact(record, snapshot)
        started = time.perf_counter()
        self._result.publish(String(data=json.dumps(card, ensure_ascii=False)))
        compact_s = time.perf_counter() - started
        record.setdefault("timing_s", {})
        record["timing_s"]["compact_publish_s"] = compact_s
        if arrived is not None:
            record["latency_to_compact_s"] = time.perf_counter() - float(arrived)
        published["ok"] = True
        with self._lock:
            self._mark_unlocked("compact")
            self._latest_result = {
                "sequence": record.get("sequence"),
                "status": record.get("status"),
                "session": record.get("session_id"),
            }
        if record.get("status") in ("processing_timeout", "processing_error"):
            self._last_success = None

    def _claim_output_dir(self) -> None:
        """One run owns the directory. A later start does not append an existing journal."""
        occupied = [
            name
            for name in ("frames.jsonl", "publish_timing.jsonl", "viz_timing.jsonl", "node_counts.json", "run.lock")
            if (self._output_dir / name).exists()
        ]
        if occupied:
            raise RuntimeError(
                "каталог результата уже занят ("
                + ", ".join(occupied)
                + "). Новый запуск не дописывает чужой журнал. Укажите пустой каталог."
            )
        lock = self._output_dir / "run.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(
                fd,
                (
                    f"pid={os.getpid()}\n"
                    "owner=obstacle_node\n"
                    "resume_not_supported=true\n"
                    "append_forbidden=true\n"
                    "key=session_id+callback_sequence\n"
                ).encode("utf-8"),
            )
            os.fsync(fd)
        finally:
            os.close(fd)

    def _enqueue_journal(self, record: dict) -> None:
        record.pop("_highlight", None)
        try:
            self._journal_q.put(record, timeout=2.0)
        except queue.Full:
            with self._lock:
                self._journal_counts["waited"] += 1
            try:
                self._journal_q.put(record, timeout=30.0)
            except queue.Full:
                with self._lock:
                    self._journal_counts["errors"] += 1
                self.get_logger().error(
                    f"journal queue stayed full for sequence {record.get('sequence')}; writing on the supervisor"
                )
                self._count_saved(self._write_journal_record(record))

    def _count_saved(self, saved: bool) -> None:
        if saved:
            with self._lock:
                self._journal_counts["written"] += 1

    def _journal_loop(self) -> None:
        while True:
            item = self._journal_q.get()
            if item is None:
                return
            try:
                saved = self._write_journal_record(item)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._journal_counts["errors"] += 1
                self.get_logger().error(f"journal write failed: {type(exc).__name__}: {exc}")
                continue
            self._count_saved(saved)

    def _write_journal_record(self, record: dict) -> bool:
        started = time.perf_counter()
        line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        timing_line = json.dumps(
            {
                "sequence": record.get("sequence"),
                "job_id": record.get("job_id"),
                "serialize_s": None,
                "compact_publish_s": (record.get("timing_s") or {}).get("compact_publish_s"),
                "latency_from_arrival_s": record.get("latency_from_arrival_s"),
                "latency_to_compact_s": record.get("latency_to_compact_s"),
                "highlight_count": record.get("highlight_count"),
            }
        ) + "\n"
        serialize_s = time.perf_counter() - started
        timing = json.loads(timing_line)
        timing["serialize_s"] = serialize_s
        timing["saved_after_flush"] = True
        timing["journal_key"] = list(journal_key(record))
        timing_line = json.dumps(timing) + "\n"
        key = journal_key(record)
        with self._journal_file_lock:
            if key in self._journal_keys:
                self._journal_counts["duplicates_rejected"] += 1
                return False
            self._jsonl.write(line)
            self._jsonl.flush()
            os.fsync(self._jsonl.fileno())
            self._timing_handle.write(timing_line)
            self._timing_handle.flush()
            os.fsync(self._timing_handle.fileno())
            self._journal_keys.add(key)
        return True

    def _offer_picture(self, record: dict, xyz=None, header=None, shm_name=None, nbytes=0, spec=None) -> bool:
        """Keep one showable cloud. Return True when the caller must not release shared memory."""
        session = int(record.get("session_id") or 0)
        reception = int(record.get("reception_index") or 0)
        status = str(record.get("status") or "")
        if not self._viz_enabled:
            return False
        with self._viz_lock:
            slot, note = picture_update(self._viz_slot, status, session, reception)
            if status != "ok":
                self._viz_note = note
                return False
            if slot is self._viz_slot:
                return False
            previous = self._viz_slot
            highlight = record.get("_highlight")
            if highlight is None:
                highlight = highlight_rows(record)
            self._viz_slot = {
                "session": session,
                "reception": reception,
                "status": "ok",
                "sequence": record.get("sequence"),
                "ready_mono": time.perf_counter(),
                "view": _viz_view(record),
                "highlight": highlight,
                "xyz": xyz,
                "header": header,
                "name": shm_name,
                "nbytes": int(nbytes),
                "spec": spec,
                "reading": False,
                "retired": False,
                "prepared": None,
            }
            if previous is not None:
                self._release_picture_slot(previous)
        with self._lock:
            self._viz_waiting = True
            if previous is not None:
                self._viz_counts["replaced"] += 1
        return shm_name is not None

    def _release_picture_slot(self, slot: dict) -> None:
        """Caller holds the picture lock. A slot still being read is released by the reader."""
        if slot.get("reading"):
            slot["retired"] = True
            return
        name = slot.get("name")
        if name and self._buffers is not None:
            self._buffers.release(name)

    def _viz_subscribers(self) -> int:
        return (
            self._display.get_subscription_count()
            + self._rviz_cloud.get_subscription_count()
            + self._markers.get_subscription_count()
            + self._cloud.get_subscription_count()
        )

    def _viz_loop(self) -> None:
        period = 1.0 / max(float(self._viz_rate), 0.1)
        next_due = time.perf_counter()
        while not self._stop.is_set():
            self._viz_take_result()
            now = time.perf_counter()
            if now < next_due:
                self._stop.wait(min(0.05, next_due - now))
                continue
            next_due = now + period
            self._viz_apply_reset()
            if not self._viz_enabled or self._viz_busy:
                continue
            try:
                self._viz_submit()
            except Exception as exc:  # noqa: BLE001
                self._viz_busy = False
                self.get_logger().error(f"visualization submit failed: {type(exc).__name__}: {exc}")

    def _viz_apply_reset(self) -> None:
        with self._lock:
            reset = self._viz_reset_session
            self._viz_reset_session = None
        if reset is None:
            return
        with self._viz_lock:
            previous = self._viz_slot
            self._viz_slot = None
            self._viz_note = None
            if previous is not None:
                self._release_picture_slot(previous)
        with self._lock:
            self._viz_waiting = False
            self._shown = None
        try:
            self._clear_previous(self._blank_header())
        except Exception:  # noqa: BLE001
            pass

    def _blank_header(self):
        from std_msgs.msg import Header

        header = Header()
        header.frame_id = "hesai_lidar"
        return header

    def _header_from_stamp(self, frame_id: str, stamp_ns: int):
        from std_msgs.msg import Header

        header = Header()
        header.frame_id = str(frame_id)
        header.stamp.sec = int(int(stamp_ns) // 1_000_000_000)
        header.stamp.nanosec = int(int(stamp_ns) % 1_000_000_000)
        return header

    def _viz_submit(self) -> None:
        if self._viz_in is None or self._viz_busy:
            return
        with self._viz_lock:
            slot = self._viz_slot
            if slot is None:
                return
            if slot.get("prepared") is not None and slot.get("prepared_generation") == slot.get("generation"):
                prepared = slot["prepared"]
                view = slot.get("view")
                ready = float(slot["ready_mono"])
                job = None
            else:
                prepared = None
                job = None
                if self._viz_subscribers() == 0:
                    pass
                else:
                    self._viz_generation += 1
                    generation = self._viz_generation
                    slot["generation"] = generation
                    slot["reading"] = True
                    view = slot.get("view")
                    ready = float(slot["ready_mono"])
                    job = {
                        "name": slot.get("name"),
                        "nbytes": int(slot.get("nbytes") or 0),
                        "spec": slot.get("spec"),
                        "highlight": slot.get("highlight"),
                        "xyz": slot.get("xyz"),
                        "limit": int(self._display_limit),
                        "generation": generation,
                        "sequence": slot.get("sequence"),
                        "frame_id": (view or {}).get("header_frame_id") or "hesai_lidar",
                        "stamp_ns": int((view or {}).get("header_stamp_ns") or 0),
                        "view": view,
                        "ready_mono": ready,
                    }
                    self._viz_inflight = {
                        "name": slot.get("name"),
                        "generation": generation,
                        "view": view,
                        "ready_mono": ready,
                    }
                    slot["name"] = None
        if prepared is not None:
            self._viz_republish(slot, prepared, view, ready)
            return
        if job is None:
            with self._lock:
                self._viz_counts["skipped_unwatched"] += 1
            return
        self._viz_in.put_nowait(job)
        self._viz_busy = True

    def _viz_take_result(self) -> None:
        if self._viz_out is None or not self._viz_busy:
            return
        try:
            result = self._viz_out.get_nowait()
        except queue.Empty:
            return
        self._viz_busy = False
        inflight = self._viz_inflight
        self._viz_inflight = None
        name = None if inflight is None else inflight.get("name")
        if name and self._buffers is not None:
            self._buffers.release(name)
        with self._viz_lock:
            slot = self._viz_slot
            if slot is not None:
                slot["reading"] = False
        if not result or not result.get("ok"):
            if result and result.get("error"):
                self.get_logger().error(f"visualization prepare failed: {result.get('error')}")
            return
        if inflight is None or int(result.get("generation") or -1) != int(inflight["generation"]):
            return
        header = self._header_from_stamp(result["frame_id"], result["stamp_ns"])
        sequence = int(result.get("sequence") or 0)
        display = _display_message(header, result["shown"], result["labels"], sequence)
        rviz = _display_message(header, result["shown"], result["labels"], sequence)
        rviz.header.frame_id = str(header.frame_id)
        prepared = {"cloud": None, "display": display, "rviz": rviz, "header": header}
        with self._viz_lock:
            if self._viz_slot is not None and int(self._viz_slot.get("generation") or -1) == int(result["generation"]):
                self._viz_slot["prepared"] = prepared
                self._viz_slot["prepared_generation"] = int(result["generation"])
                slot = self._viz_slot
            else:
                slot = None
        if slot is None:
            return
        with self._lock:
            self._mark_unlocked("viz_prepared", "prepared")
        self._viz_send(slot, result["view"], float(result["ready_mono"]), float(result["prepare_s"]), prepared=prepared)

    def _viz_tick(self) -> None:
        return
        with self._viz_lock:
            slot = self._viz_slot
            if slot is None:
                return
            prepared = slot.get("prepared")
            if prepared is None:
                name = slot.get("name")
                nbytes = int(slot.get("nbytes") or 0)
                xyz = slot.get("xyz")
                spec = slot.get("spec")
                header = slot.get("header")
                highlight = slot.get("highlight")
                view = slot.get("view")
                ready = float(slot["ready_mono"])
            else:
                name = None
                xyz = None
                spec = None
                header = None
                highlight = None
                view = slot.get("view")
                ready = float(slot["ready_mono"])
                nbytes = 0
        if prepared is not None:
            self._viz_republish(slot, prepared, view, ready)
            return
        if self._viz_subscribers() == 0:
            with self._lock:
                self._viz_counts["skipped_unwatched"] += 1
            return
        with self._viz_lock:
            if self._viz_slot is not slot:
                return
            slot["reading"] = True
        prepare_started = time.perf_counter()
        try:
            cloud_msg, display, rviz, header = self._prepare_picture(xyz, name, nbytes, spec, header, highlight, view)
        finally:
            with self._viz_lock:
                slot["reading"] = False
                stale = bool(slot.get("retired")) or self._viz_slot is not slot
                name_held = slot.get("name")
                slot["name"] = None
            if name_held and self._buffers is not None:
                self._buffers.release(name_held)
        if stale or display is None:
            return
        prepare_s = time.perf_counter() - prepare_started
        with self._viz_lock:
            if self._viz_slot is slot:
                slot["prepared"] = {"cloud": cloud_msg, "display": display, "rviz": rviz, "header": header}
        with self._lock:
            self._mark_unlocked("viz_prepared", "prepared")
        self._viz_send(slot, view, ready, prepare_s)

    def _prepare_picture(self, xyz, name, nbytes, spec, header, highlight, view):
        from std_msgs.msg import Header

        if xyz is None:
            if not name:
                return None, None, None, header
            from multiprocessing import shared_memory

            shm = shared_memory.SharedMemory(name=name)
            try:
                raw = np.ndarray((int(nbytes),), dtype=np.uint8, buffer=shm.buf).copy()
            finally:
                shm.close()
            decoded = decode_recorded(raw, spec)
            xyz = np.column_stack((decoded["x"], decoded["y"], decoded["z"])).astype(np.float32, copy=False)
            header = Header()
            header.frame_id = str(spec["frame_id"])
            header.stamp.sec = int(spec["stamp_ns"] // 1_000_000_000)
            header.stamp.nanosec = int(spec["stamp_ns"] % 1_000_000_000)
            cloud_msg = self._message_from_raw(spec, raw)
        else:
            cloud_msg = None
        shown, labels = sample_display(xyz, highlight, self._display_limit)
        sequence = int(view.get("sequence") or 0)
        display = _display_message(header, shown, labels, sequence)
        rviz = _display_message(header, shown, labels, sequence)
        rviz.header.frame_id = str(header.frame_id)
        return cloud_msg, display, rviz, header

    def _viz_republish(self, slot, prepared, view, ready) -> None:
        if self._viz_subscribers() == 0:
            return
        self._viz_send(slot, view, ready, prepare_s=0.0, republish=True, prepared=prepared)

    def _viz_send(self, slot, view, ready, prepare_s: float, republish: bool = False, prepared=None) -> None:
        with self._viz_lock:
            current = slot.get("prepared") if prepared is None else prepared
            if current is None or self._viz_slot is not slot:
                return
            cloud_msg = current.get("cloud")
            display = current.get("display")
            rviz = current.get("rviz")
            header = current.get("header") or self._blank_header()
        with self._lock:
            latest = dict(self._latest_result) if self._latest_result else None
        age = max(0.0, time.perf_counter() - ready)
        text = _picture_text(view, age, latest)
        publish_started = time.perf_counter()
        if cloud_msg is not None and self._cloud.get_subscription_count():
            self._cloud.publish(cloud_msg)
        if display is not None:
            self._display.publish(display)
            self._rviz_cloud.publish(rviz)
        self._markers.publish(_markers(view, header, text))
        publish_s = time.perf_counter() - publish_started
        with self._lock:
            if republish:
                self._viz_counts["redrawn"] = int(self._viz_counts.get("redrawn", 0)) + 1
            else:
                self._mark_unlocked("viz_published", "published")
            self._shown = {
                "sequence": view.get("sequence"),
                "status": view.get("status"),
                "session": view.get("session_id"),
                "age_s": round(age, 3),
            }
        try:
            with self._journal_file_lock:
                self._viz_timing.write(
                    json.dumps(
                        {
                            "sequence": view.get("sequence"),
                            "prepare_s": prepare_s,
                            "publish_s": publish_s,
                            "republish": republish,
                            "age_s": round(age, 3),
                        }
                    )
                    + "\n"
                )
                self._viz_timing.flush()
        except OSError:
            pass

    def _clear_previous(self, header) -> None:
        array = MarkerArray()
        array.markers.append(_clear_marker(header))
        self._previous_markers.publish(array)

    def _one_frame(self, msg: PointCloud2, sequence: int) -> tuple[dict, np.ndarray]:
        decode_started = time.perf_counter()
        decoded = decode_message(msg)
        decode_s = time.perf_counter() - decode_started
        xyz = np.column_stack((decoded["x"], decoded["y"], decoded["z"])).astype(np.float32, copy=False)
        group_started = time.perf_counter()
        valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
        groups = group_exact_xyz(
            decoded["x"], decoded["y"], decoded["z"], decoded["intensity"], decoded["ring"], decoded["timestamp"], valid
        )
        group_s = time.perf_counter() - group_started
        progress = f"/tmp/obstacle_progress_{int(sequence)}.json"
        built = call_bounded(
            execute_grouped_frame,
            (
                {
                    "groups": groups,
                    "corridor_config": self._corridor,
                    "detector_config": self._detector,
                    "mount": self._mount,
                    "flags": self._flags,
                    "stamp_ns": decoded["stamp_ns"],
                    "frame_id": decoded["frame_id"],
                    "message_index": int(sequence),
                    "straight_profile": self._straight,
                },
            ),
            self._budget,
            progress,
        )
        worker_s = built.get("worker_elapsed_s", built.get("elapsed_s"))
        if built.get("status") == "processing_timeout":
            built = {
                "status": "processing_timeout",
                "decision": "processing_timeout",
                "header_stamp_ns": int(decoded["stamp_ns"]),
                "header_frame_id": decoded["frame_id"],
                "candidates": None,
                "conditional_intrusions": [],
                "absence_of_candidates_is_not_clear": True,
                "empty_result_is_not_clear": True,
                "cluster_progress": built.get("cluster_progress"),
                "worker_alive_after_stop": built.get("worker_alive_after_stop"),
                "corridor_source": self._source,
                "config_version": config_version(self._detector, self._corridor, self._mount, self._flags, None),
                "algorithm_version": algorithm_version(),
                "intrusion_rule_present": True,
                "intrusion_rule_reason": "frame_did_not_finish",
                "timing_s": {},
            }
        elif built.get("status") == "processing_error":
            built["decision"] = "processing_error"
            built["candidates"] = None
            built["conditional_intrusions"] = []
        built.setdefault("timing_s", {})
        built["timing_s"]["decode"] = decode_s
        built["timing_s"]["group"] = group_s
        built["timing_s"]["worker_and_publish_prep"] = worker_s
        return built, xyz

    def _submit_stream(self, item) -> None:
        msg, reception, session, stamp, _token = item
        sequence = int(reception - 1)
        arrived = self._arrived.get(reception, time.perf_counter())
        shm_name, nbytes = self._buffers.write(bytes(msg.data))
        job_id = self._next_job
        self._next_job += 1
        spec = {
            "job_id": job_id,
            "reception": int(reception),
            "session": int(session),
            "stamp_ns": int(stamp),
            "frame_id": str(msg.header.frame_id),
            "message_index": sequence,
            "shm_name": shm_name,
            "nbytes": int(nbytes),
            "height": int(msg.height),
            "width": int(msg.width),
            "point_step": int(msg.point_step),
            "row_step": int(msg.row_step),
            "is_bigendian": bool(msg.is_bigendian),
            "fields": [(field.name, int(field.offset), int(field.datatype), int(field.count)) for field in msg.fields],
            "corridor_config": self._corridor,
            "detector_config": self._detector,
            "mount": self._mount,
            "flags": self._flags,
            "straight_profile": self._straight,
            "config_version": config_version(self._detector, self._corridor, self._mount, self._flags, None),
        }
        self._jobs[job_id] = {"spec": spec, "arrived": arrived, "started": time.perf_counter()}
        self._deadlines[job_id] = arrived + float(self._budget)
        self._pool.submit(job_id, process_stream_job, (spec,), str(self._output_dir / f"progress_{job_id}.json"))
        with self._lock:
            self._counts["in_flight"] = sum(slot["job_id"] is not None for slot in self._pool.slots)

    def _keep_scene(self, session: int, reception: int) -> bool:
        if session < self._session_id and self._shown_reception.get(self._session_id, -1) >= 0:
            return False
        if reception < self._shown_reception.get(session, -1):
            return False
        self._shown_reception[session] = int(reception)
        return True

    def _message_from_raw(self, spec: dict, raw: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = str(spec["frame_id"])
        msg.header.stamp.sec = int(spec["stamp_ns"] // 1_000_000_000)
        msg.header.stamp.nanosec = int(spec["stamp_ns"] % 1_000_000_000)
        msg.height = int(spec["height"])
        msg.width = int(spec["width"])
        msg.point_step = int(spec["point_step"])
        msg.row_step = int(spec["row_step"])
        msg.is_bigendian = bool(spec["is_bigendian"])
        msg.is_dense = False
        msg.fields = [
            PointField(name=name, offset=int(offset), datatype=int(datatype), count=int(count))
            for name, offset, datatype, count in spec["fields"]
        ]
        msg.data = raw.tobytes()
        return msg

    def _collect_stream(self) -> None:
        for record in self._pool.poll(time.perf_counter(), self._deadlines):
            job_id = int(record["job_id"])
            meta = self._jobs.pop(job_id, None)
            self._deadlines.pop(job_id, None)
            if meta is None:
                continue
            spec = meta["spec"]
            now = time.perf_counter()
            record["playback_mode"] = "stream"
            record["player"] = "ros2_bag_play"
            record["run_id"] = None
            record["bag_time_ns"] = None
            record["sequence"] = int(spec["message_index"])
            record["message_index"] = int(spec["message_index"])
            record["callback_sequence"] = int(spec["message_index"])
            record["message_index_is_callback_ordinal"] = True
            record["message_index_is_not_a_bag_index"] = True
            record["frame_index"] = int(spec["message_index"])
            record["result_elapsed_s"] = now - float(meta["arrived"])
            record["latency_from_arrival_s"] = record["result_elapsed_s"]
            record.setdefault("timing_s", {})
            record["timing_s"]["queue_wait_s"] = float(meta["started"]) - float(meta["arrived"])
            record["budget_is_artificial_timeout"] = self._artificial_budget
            record["internal_format_is_not_a_contest_api"] = True
            record["algorithm_version"] = algorithm_version()
            record["config_version"] = spec.get("config_version") or record.get("config_version")
            record["job_id"] = f"{int(spec['session'])}:{int(spec['message_index'])}"
            record["reception_index"] = int(spec["reception"])
            record["session_id"] = int(spec["session"])
            record["header_stamp_ns"] = int(spec["stamp_ns"])
            record["header_frame_id"] = str(spec["frame_id"])
            record["timing_s"]["submit_to_collect_s"] = now - float(meta["started"])
            self._arrived.pop(int(spec["reception"]), None)
            with self._lock:
                self._counts["in_flight"] = sum(slot["job_id"] is not None for slot in self._pool.slots)
                status = record.get("status")
                if status == "processing_timeout":
                    self._counts["processing_timeout"] += 1
                elif status == "processing_error":
                    self._counts["processing_error"] += 1
                else:
                    self._counts["completed_ok"] += 1
                self._mark_unlocked("completed")
                self._processing_id += 1
                processing_id = self._processing_id
                received_now = self._counts["received"]
                current = self._reception
                snapshot = self._snapshot()
            annotate_display(
                record,
                processing_id=processing_id,
                session_id=int(spec["session"]),
                stamp_ns=int(spec["stamp_ns"]),
                frame_id=spec["frame_id"],
                superseded=int(spec["reception"]) != int(current),
                received_now=received_now,
                reception_index=int(spec["reception"]),
            )
            record["result_is_current"] = int(spec["reception"]) == int(current)
            record["result_is_answer_to_latest_input"] = record["result_is_current"]
            record["late_result_not_shown"] = False
            published = {"ok": False, "counted": True}
            self._displayed_session = int(spec["session"])
            self._publish_compact(record, snapshot, published, meta["arrived"])
            picture_spec = {
                "frame_id": spec["frame_id"],
                "stamp_ns": spec["stamp_ns"],
                "height": spec["height"],
                "width": spec["width"],
                "point_step": spec["point_step"],
                "row_step": spec["row_step"],
                "is_bigendian": spec["is_bigendian"],
                "fields": spec["fields"],
            }
            retained = self._offer_picture(
                record,
                shm_name=spec["shm_name"],
                nbytes=int(spec["nbytes"]),
                spec=picture_spec,
            )
            record["late_result_not_shown"] = bool(self._viz_enabled and record.get("status") == "ok" and not retained)
            if not retained:
                self._buffers.release(spec["shm_name"])
            self._enqueue_journal(record)

    def destroy_node(self) -> None:
        self._stop.set()
        if getattr(self, "_thread", None) is not None:
            self._thread.join(timeout=15.0)
        if getattr(self, "_viz_thread", None) is not None:
            self._viz_thread.join(timeout=5.0)
        if getattr(self, "_viz_in", None) is not None:
            try:
                self._viz_in.put(None, timeout=1.0)
            except queue.Full:
                pass
        if getattr(self, "_viz_process", None) is not None:
            self._viz_process.join(timeout=5.0)
            if self._viz_process.is_alive():
                self._viz_process.terminate()
        if getattr(self, "_journal_q", None) is not None:
            try:
                self._journal_q.put(None, timeout=5.0)
            except queue.Full:
                pass
        if getattr(self, "_journal_thread", None) is not None:
            self._journal_thread.join(timeout=15.0)
            if not self._journal_thread.is_alive():
                with self._lock:
                    self._journal_counts["closed"] = 1
        try:
            for handle_name in ("_jsonl", "_timing_handle", "_viz_timing"):
                handle = getattr(self, handle_name, None)
                if handle is None:
                    continue
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            with self._lock:
                self._journal_counts["closed"] = 0
                self._journal_counts["errors"] += 1
        try:
            with self._lock:
                payload = self._snapshot()
            (self._output_dir / "node_counts.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        for handle_name in ("_jsonl", "_timing_handle", "_viz_timing"):
            handle = getattr(self, handle_name, None)
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass
        if self._pool is not None:
            self._pool.shutdown()
        if self._buffers is not None:
            self._buffers.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
