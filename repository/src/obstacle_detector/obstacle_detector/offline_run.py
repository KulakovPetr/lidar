"""Sequential offline processing of one bag. No live node.

The packaged detector, geometry and thresholds stay in the image.
`--config` supplies only the mount. KISS-ICP and accumulated context stay off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from obstacle_detector.bag_io import decode_message, open_reader
from obstacle_detector.batch import append_record, read_completed, resume_is_compatible
from obstacle_detector.corridor import APPLIED_MOUNT_SOURCES, load_config, mount_from_config
from obstacle_detector.frame_budget import call_bounded
from obstacle_detector.frame_pipeline import algorithm_version, config_dir, load_runtime, variant_flags
from obstacle_detector.repeat_groups import group_exact_xyz, valid_mask
from obstacle_detector.working import config_version

KNOWN_TOPICS = ("/lidar_points", "/sensing/lidar/hesai128/pointcloud")
MOUNT_KEYS = {"confirmed", "source", "status", "basis", "translation_m", "rpy_rad"}
ANGLE_KEYS = ("roll", "pitch", "yaw")
AXIS_KEYS = ("x", "y", "z")


def select_topic(topics, explicit: str | None) -> tuple[str, str]:
    """Use the only PointCloud2. Several topics require --topic and are not merged."""
    names = [topic.name for topic in topics if topic.type == "sensor_msgs/msg/PointCloud2"]
    if explicit:
        if explicit not in names:
            raise SystemExit(f"топик {explicit} не является PointCloud2 в этом bag: {names}")
        return explicit, "явный выбор"
    if len(names) == 1:
        return names[0], "единственный PointCloud2"
    if not names:
        raise SystemExit("в bag нет топика PointCloud2")
    raise SystemExit("несколько топиков PointCloud2, нужен --topic; потоки не объединяются: " + ", ".join(names))


def content_fingerprint(bag_dir: Path) -> str:
    """Hash sqlite storage bytes. The directory path is not part of the fingerprint."""
    files = sorted(path for path in bag_dir.glob("*.db3") if path.is_file())
    if not files:
        raise SystemExit(f"в {bag_dir} нет файла .db3")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hash.update(chunk)
        digest.update(file_hash.digest())
    return digest.hexdigest()[:16]


def load_mount_file(path: Path) -> tuple[dict, dict]:
    """Accept a mount file. Zeros do not become a confirmed calibration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"mount"}:
        raise SystemExit("конфигурация поддерживает только блок mount")
    block = raw["mount"]
    if not isinstance(block, dict) or set(block) != MOUNT_KEYS:
        raise SystemExit("неподдерживаемая конфигурация установки")
    if set(block["translation_m"]) != set(AXIS_KEYS) or set(block["rpy_rad"]) != set(ANGLE_KEYS):
        raise SystemExit("неподдерживаемая конфигурация установки")
    numbers = [float(block["translation_m"][key]) for key in AXIS_KEYS]
    numbers.extend(float(block["rpy_rad"][key]) for key in ANGLE_KEYS)
    confirmed = block.get("confirmed")
    if confirmed is True:
        if all(value == 0.0 for value in numbers):
            raise SystemExit("нулевая установка не является калибровкой")
        if block.get("source") not in APPLIED_MOUNT_SOURCES or not str(block.get("basis") or "").strip():
            raise SystemExit("неподдерживаемая конфигурация установки")
    elif confirmed is not False:
        raise SystemExit("неподдерживаемая конфигурация установки")
    corridor = load_config(config_dir() / "corridor.yaml")
    corridor["mount"] = block
    mount = mount_from_config(corridor)
    if confirmed is False and mount["applied"]:
        raise SystemExit("неподтверждённая установка не применяется")
    return corridor, mount


def reader_version() -> str:
    """Hash of the bag reader. Separate from the detector math version."""
    digest = hashlib.sha256()
    digest.update(Path(__file__).with_name("bag_io.py").read_bytes())
    return digest.hexdigest()[:16]


def run_identity(version: str, effective: str, topic: str, fingerprint: str, reader: str, corridor_source: str = "inferred", profile_bytes: bytes = b"", frame_budget_s: float = 30.0) -> str:
    """Hash core, reader, mount, detector, profile bytes, budget and bag bytes.

    A changed profile or budget must not resume into an older result file.
    """
    digest = hashlib.sha256()
    digest.update(version.encode())
    digest.update(reader.encode())
    digest.update(effective.encode())
    digest.update(topic.encode())
    digest.update(fingerprint.encode())
    digest.update(corridor_source.encode())
    digest.update(profile_bytes)
    digest.update(f"{float(frame_budget_s):.6f}".encode())
    return digest.hexdigest()[:16]


def _machine() -> dict:
    cpu = "unknown"
    mem = None
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        mem = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").split()[1]
    except OSError:
        pass
    return {"cpu": cpu, "meminfo_kib": mem, "not_part_of_run_identity": True}


def exit_code(summary: dict) -> int:
    """A frame that failed processing is not a successful run."""
    return 1 if int(summary.get("errors") or 0) else 0


def execute_grouped_frame(payload: dict) -> dict:
    """Module-level worker entry. The parent can terminate this process."""
    from obstacle_detector.working import process_working_frame

    return process_working_frame(
        payload["groups"],
        payload["corridor_config"],
        payload["detector_config"],
        payload["mount"],
        payload["flags"],
        stamp_ns=payload["stamp_ns"],
        header_frame_id=payload["frame_id"],
        message_index=payload["message_index"],
        frame_id=payload["frame_id"],
        frame_key=payload["message_index"],
        straight_profile=payload.get("straight_profile"),
    )


def process_stream_job(job: dict) -> dict:
    """Decode, group and detect one cloud inside a pool worker.

    The shared buffer is copied immediately and then detached, so the parent
    can reuse it only after this process has stopped reading it.
    """
    import time
    from multiprocessing import shared_memory

    import numpy as np

    from obstacle_detector.bag_io import decode_recorded
    from obstacle_detector.repeat_groups import group_exact_xyz, valid_mask

    started = time.perf_counter()
    shm = shared_memory.SharedMemory(name=job["shm_name"])
    try:
        raw = np.ndarray((int(job["nbytes"]),), dtype=np.uint8, buffer=shm.buf).copy()
    finally:
        shm.close()
    decoded = decode_recorded(raw, job)
    valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
    groups = group_exact_xyz(
        decoded["x"], decoded["y"], decoded["z"], decoded["intensity"], decoded["ring"], decoded["timestamp"], valid
    )
    prepared = time.perf_counter()
    result = execute_grouped_frame(
        {
            "groups": groups,
            "corridor_config": job["corridor_config"],
            "detector_config": job["detector_config"],
            "mount": job["mount"],
            "flags": job["flags"],
            "stamp_ns": int(job["stamp_ns"]),
            "frame_id": str(job["frame_id"]),
            "message_index": int(job["message_index"]),
            "straight_profile": job.get("straight_profile"),
        }
    )
    result["job_id"] = int(job["job_id"])
    result["reception_index"] = int(job["reception"])
    result["session_id"] = int(job["session"])
    result["header_stamp_ns"] = int(job["stamp_ns"])
    result["header_frame_id"] = str(job["frame_id"])
    result["frame_index"] = int(job["message_index"])
    timing = dict(result.get("timing_s") or {})
    timing["prepare_s"] = prepared - started
    result["timing_s"] = timing
    result["config_version"] = job.get("config_version")
    return result


def process_bag(bag_dir: Path, output_dir: Path, config_path: Path, topic: str | None, start: int, count: int | None, corridor_source: str = "inferred", profile_path: Path | None = None, frame_budget_s: float = 30.0) -> dict:
    if start < 0:
        raise SystemExit("--start должен быть неотрицательным")
    if count is not None and count < 0:
        raise SystemExit("--count должен быть неотрицательным")
    corridor_config, mount = load_mount_file(config_path)
    _packaged_corridor, detector_config, batch_config, _packaged_mount = load_runtime()
    flags = variant_flags(batch_config, batch_config["default_variant"])
    if flags["name"] != "C" or flags["connectivity"] != "distance" or not flags["local_protrusions_enabled"]:
        raise SystemExit("поставка запускает только вариант C")
    version = algorithm_version()
    reader_hash = reader_version()
    effective = config_version(detector_config, corridor_config, mount, flags, None)
    fingerprint = content_fingerprint(bag_dir)
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    reader = open_reader(bag_dir)
    chosen, topic_reason = select_topic(list(reader.get_all_topics_and_types()), topic)
    if corridor_source not in ("inferred", "configured_straight"):
        raise SystemExit("corridor source должен быть inferred или configured_straight")
    straight_profile = None
    profile_bytes = b""
    profile_raw = None
    if corridor_source == "configured_straight":
        if profile_path is None:
            raise SystemExit("configured_straight требует --profile")
        from obstacle_detector.straight_corridor import load_profile

        profile_bytes = Path(profile_path).read_bytes()
        profile_raw = yaml.safe_load(profile_bytes)
        straight_profile = load_profile(profile_path)
    identity = run_identity(
        version, effective, chosen, fingerprint, reader_hash, corridor_source, profile_bytes, frame_budget_s
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "frames.jsonl"
    completed = read_completed(jsonl)
    resume_is_compatible(completed, identity)
    manifest = {
        "algorithm_version": version,
        "reader_version": reader_hash,
        "config_version": effective,
        "run_identity": identity,
        "topic": chosen,
        "topic_reason": topic_reason,
        "content_fingerprint": fingerprint,
        "bag_name_not_used_in_identity": bag_dir.name,
        "variant": flags,
        "kiss_icp": False,
        "local_context": False,
        "mount_applied": bool(mount.get("applied")),
        "coordinate_frame_when_mount_unconfirmed": "sensor_working_prior_mount_unconfirmed",
        "range_from_train_nose_m": None,
        "known_topics": list(KNOWN_TOPICS),
        "live_mode_not_included": True,
        "corridor_source": corridor_source,
        "frame_budget_s": float(frame_budget_s),
        "frame_budget_is_not_a_detection_threshold": True,
        "configured_straight_enabled_by_default": False,
        "effective_configuration": {
            "detector": detector_config,
            "corridor": corridor_config,
            "mount": mount,
            "variant": flags,
            "corridor_source": corridor_source,
            "profile": profile_raw,
            "frame_budget_s": float(frame_budget_s),
            "intrusion_rule": "configured_volume_unexplained_cluster",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "machine.json").write_text(json.dumps(_machine(), ensure_ascii=False, indent=2), encoding="utf-8")
    seen = 0
    processed = 0
    errors = 0
    timeouts = 0
    started_at = time.perf_counter()
    while reader.has_next():
        topic_name, payload, record_ns = reader.read_next()
        if topic_name != chosen:
            continue
        index = seen
        seen += 1
        if index < start:
            continue
        if count is not None and index >= start + count:
            break
        if index in completed:
            continue
        record = {
            "frame_index": index,
            "message_index": index,
            "algorithm_version": version,
            "reader_version": reader_hash,
            "config_version": effective,
            "run_identity": identity,
            "content_fingerprint": fingerprint,
            "bag_record_time_ns": int(record_ns),
        }
        t0 = time.perf_counter()
        try:
            decode_started = time.perf_counter()
            decoded = decode_message(deserialize_message(payload, PointCloud2))
            decode_s = time.perf_counter() - decode_started
            group_started = time.perf_counter()
            valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
            groups = group_exact_xyz(
                decoded["x"], decoded["y"], decoded["z"], decoded["intensity"], decoded["ring"], decoded["timestamp"], valid
            )
            group_s = time.perf_counter() - group_started
            progress = str(output_dir / f"progress_{index}.json")
            built = call_bounded(
                execute_grouped_frame,
                (
                    {
                        "groups": groups,
                        "corridor_config": corridor_config,
                        "detector_config": detector_config,
                        "mount": mount,
                        "flags": flags,
                        "stamp_ns": decoded["stamp_ns"],
                        "frame_id": decoded["frame_id"],
                        "message_index": index,
                        "straight_profile": straight_profile,
                    },
                ),
                frame_budget_s,
                progress,
            )
            if built.get("status") in ("processing_timeout", "processing_error"):
                record["status"] = built["status"]
                record["header_stamp_ns"] = int(decoded["stamp_ns"])
                record["header_frame_id"] = decoded["frame_id"]
                record["candidates"] = None
                record["conditional_intrusions"] = []
                record["decision"] = built["status"]
                record["absence_of_candidates_is_not_clear"] = True
                record["empty_result_is_not_clear"] = True
                record["obstacle_reported"] = None
                record["cluster_progress"] = built.get("cluster_progress")
                record["error"] = built.get("error")
                record["worker_alive_after_stop"] = built.get("worker_alive_after_stop")
                record["timing_s"] = {"decode": decode_s, "group": group_s, "worker": built.get("elapsed_s")}
                if built["status"] == "processing_error":
                    errors += 1
                else:
                    timeouts += 1
            else:
                record.update(built)
                record.setdefault("timing_s", {})
                record["timing_s"]["decode"] = decode_s
                record["timing_s"]["group"] = group_s
        except Exception as exc:  # noqa: BLE001
            record["status"] = "processing_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["candidates"] = None
            record["absence_of_candidates_is_not_clear"] = True
            record["empty_result_is_not_clear"] = True
            errors += 1
        record.setdefault("timing_s", {})["frame_excluding_bag_read"] = time.perf_counter() - t0
        append_record(jsonl, record)
        processed += 1
    return {
        "messages_seen": seen,
        "processed_now": processed,
        "already_completed": len(completed),
        "errors": errors,
        "timeouts": timeouts,
        "elapsed_s": time.perf_counter() - started_at,
        "run_identity": identity,
        "topic": chosen,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Последовательная offline-обработка одного bag")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--corridor-source", default="inferred", choices=("inferred", "configured_straight"))
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--frame-budget-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 0:
        raise SystemExit("--count должен быть неотрицательным")
    if args.frame_budget_s <= 0:
        raise SystemExit("--frame-budget-s должен быть положительным")
    summary = process_bag(
        args.bag, args.output, args.config, args.topic, args.start, args.count,
        corridor_source=args.corridor_source, profile_path=args.profile, frame_budget_s=args.frame_budget_s,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
