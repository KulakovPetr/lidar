"""Streaming batch launch. One cloud at a time. No cross-frame state.

Completed JSONL lines are not rewritten on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np

from obstacle_detector.bag_io import choose_pointcloud_topic, decode_message, open_reader, read_metadata
from obstacle_detector.frame_pipeline import algorithm_version, load_runtime, process_arrays, variant_flags
from obstacle_detector.repeat_groups import valid_mask
from obstacle_detector.view_selection import load_run_roles


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        number = value.item()
        if isinstance(number, float) and not np.isfinite(number):
            return None
        return number
    return value


def run_identity(version: str, config_dir: Path, variant: str, topic: str, bag_dir: Path, metadata: dict) -> str:
    """Fingerprint of code, loaded configs, variant, topic, and this bag."""
    digest = hashlib.sha256()
    digest.update(version.encode())
    digest.update(variant.encode())
    digest.update(topic.encode())
    digest.update(bag_dir.name.encode())
    for name in ("detector.yaml", "corridor.yaml", "batch.yaml", "pipeline.yaml", "session.yaml"):
        file_path = config_dir / name
        if name == "session.yaml" and not file_path.is_file():
            continue
        digest.update(name.encode())
        digest.update(file_path.read_bytes())
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def resume_is_compatible(completed: dict, identity: str) -> None:
    if not completed:
        return
    found = {item.get("run_identity") for item in completed.values()}
    if found != {identity}:
        raise SystemExit(
            "возобновление отказано: отпечаток запуска не совпадает "
            f"(в файле {sorted(str(item) for item in found)}, сейчас {identity}). "
            "Старые строки не переписываются и не смешиваются с другой конфигурацией."
        )


def read_completed(path: Path) -> dict[int, dict]:
    found = {}
    if not path.is_file():
        return found
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0]
        if text:
            text += "\n"
        path.write_text(text, encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found[int(record["frame_index"])] = record
    return found


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        handle.flush()


def _expected_count(metadata: dict, topic: str) -> int | None:
    for item in metadata.get("topics") or []:
        if item.get("name") == topic and item.get("type") == "sensor_msgs/msg/PointCloud2":
            return int(item["message_count"])
    return None


def _hardware() -> str:
    try:
        cpu = "unknown"
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        mem = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").split()[1]
        return f"{cpu}; память {mem} КиБ"
    except OSError:
        return "unknown"


def run_bag(bag_dir: Path, output_dir: Path, config_dir: Path, variant: str, roles: dict[str, str], topic: str | None, max_frames: int | None, message_indices: set[int] | None, from_ns: int | None, to_ns: int | None, pictures: bool, pictures_only: bool = False) -> dict:
    run_name = bag_dir.name
    role = roles.get(run_name)
    if role != "development":
        raise SystemExit(
            f"{run_name}: роль {role!r}. Полный набор на диске не разрешает обработку. "
            "Неизвестную запись сначала инвентаризируйте, без подбора параметров."
        )
    corridor_config, detector_config, batch_config, mount = load_runtime(config_dir)
    flags = variant_flags(batch_config, variant)
    version = algorithm_version()
    metadata = read_metadata(bag_dir)
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    reader = open_reader(bag_dir)
    topics = list(reader.get_all_topics_and_types())
    chosen, topic_reason = choose_pointcloud_topic(topics, batch_config["known_pointcloud_topics"], topic)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "frames.jsonl"
    completed = read_completed(jsonl)
    identity = run_identity(version, config_dir, variant, chosen, bag_dir, metadata)
    resume_is_compatible(completed, identity)
    picture_indices = {
        int(item["message_index"])
        for item in batch_config.get("picture_frames") or []
        if item.get("run") == run_name
    }
    prior = batch_config["coordinate_prior"]
    meta = {
        "run": run_name,
        "role": role,
        "algorithm_version": version,
        "run_identity": identity,
        "variant": variant,
        "connectivity": flags["connectivity"],
        "local_protrusions": flags["local_protrusions_enabled"],
        "topic": chosen,
        "topic_reason": topic_reason,
        "coordinate_prior": prior,
        "no_cross_frame_state": True,
        "resume_does_not_rewrite_completed_frames": True,
        "expected_messages": _expected_count(metadata, chosen),
        "hardware": _hardware(),
        "mount_applied": bool(mount.get("applied")),
    }
    if not pictures_only:
        (output_dir / "run_meta.json").write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    seen = 0
    processed_now = 0
    resumed = 0
    errors = 0
    time_filtered = 0
    started = time.perf_counter()
    while reader.has_next():
        topic_name, payload, record_ns = reader.read_next()
        if topic_name != chosen:
            del payload
            continue
        index = seen
        seen += 1
        wanted_picture = (pictures or pictures_only) and (
            index in picture_indices or (message_indices is not None and index in message_indices)
        )
        if message_indices is not None and index not in message_indices:
            del payload
            continue
        if pictures_only and index not in picture_indices and (message_indices is None or index not in message_indices):
            del payload
            continue
        if index in completed and not (pictures_only and wanted_picture):
            resumed += 1
            del payload
            continue
        if max_frames is not None and processed_now >= max_frames and not pictures_only:
            del payload
            break
        record = {
            "frame_index": index,
            "algorithm_version": version,
            "run_identity": identity,
            "variant": variant,
            "bag_record_time_ns": int(record_ns),
            "coordinate_prior": prior,
        }
        try:
            decode_started = time.perf_counter()
            message = deserialize_message(payload, PointCloud2)
            decoded = decode_message(message)
            decode_s = time.perf_counter() - decode_started
            del message
            stamp = int(decoded["stamp_ns"])
            if (from_ns is not None and stamp < from_ns) or (to_ns is not None and stamp > to_ns):
                time_filtered += 1
                del payload
                del decoded
                continue
            valid = valid_mask(decoded["x"], decoded["y"], decoded["z"])
            result = process_arrays(
                decoded["x"],
                decoded["y"],
                decoded["z"],
                decoded["intensity"],
                decoded["ring"],
                decoded["timestamp"],
                valid,
                corridor_config,
                detector_config,
                mount,
                flags,
                keep_plot=bool(wanted_picture or pictures_only),
            )
            plot_rows = result.pop("_plot_rows", None)
            result.pop("_plot", None)
            record.update(result)
            record["frame_id"] = decoded["frame_id"]
            record["stamp_ns"] = stamp
            record["timing_s"]["decode"] = decode_s
            record["valid_points"] = int(valid.sum())
            record["width"] = decoded["width"]
            if wanted_picture or pictures_only:
                _save_picture(output_dir / f"frame_{index}.png", decoded, result, plot_rows, run_name, index)
            del decoded
        except Exception as exc:  # noqa: BLE001 — one bad cloud must not drop the rest of the stream
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["hypotheses"] = []
            record["candidates"] = []
            record["obstacle_reported"] = None
            record["absence_of_candidates_is_not_clear"] = True
            errors += 1
        del payload
        if pictures_only:
            continue
        append_record(jsonl, record)
        completed[index] = record
        processed_now += 1
        if processed_now % 10 == 0:
            print(f"{run_name} frame {index} status {record.get('status')} new {processed_now}", flush=True)
    wall = time.perf_counter() - started
    if pictures_only:
        print(f"{run_name} pictures only, jsonl unchanged", flush=True)
        return {"run": run_name, "pictures_only": True}
    previous_wall = 0.0
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous_wall = float(json.loads(summary_path.read_text(encoding="utf-8")).get("wall_s_total") or 0.0)
    summary = summarize_run(output_dir)
    summary["wall_s_this_process"] = wall
    summary["wall_s_total"] = previous_wall + wall
    summary["processed_this_process"] = processed_now
    summary["resumed_without_recompute"] = resumed
    summary["time_filtered"] = time_filtered
    summary["errors_this_process"] = errors
    summary["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if max_frames is not None and processed_now >= max_frames:
        summary["stopped_because"] = "max_frames"
    elif message_indices is not None:
        summary["stopped_because"] = "explicit_message_indices"
    else:
        summary["stopped_because"] = "bag_end"
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{run_name} done new {processed_now} resumed {resumed} seen {seen} status_counts {summary.get('status_counts')}",
        flush=True,
    )
    return summary


def _save_picture(path: Path, decoded, result, plot_rows, run_name: str, index: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from obstacle_detector.corridor import sensor_to_working

    s, l, _h = sensor_to_working(decoded["x"], decoded["y"], decoded["z"])
    finite = np.isfinite(s) & np.isfinite(l)
    show = np.flatnonzero(finite)
    if show.size > 8000:
        show = np.sort(np.random.default_rng(4).choice(show, 8000, replace=False))
    best = None
    for item in plot_rows or []:
        if best is None or item["count"] > best["count"]:
            best = item
    hypothesis_id = None if best is None else best["hypothesis_id"]
    count = 0 if best is None else best["count"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.scatter(s[show], l[show], s=1, c="#c5c9ce", linewidths=0)
    if best is not None:
        ax.scatter(best["s"], best["l"], s=8, c="#c0392b", linewidths=0)
    ax.set_xlabel("s, м")
    ax.set_ylabel("l, м")
    ax.set_title(
        f"{run_name} кадр {index} {result.get('status')} гипотеза {hypothesis_id}: {count} точек кандидатов этой гипотезы. "
        "Не проверены и не названы ложными тревогами.",
        fontsize=8,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize_run(output_dir: Path) -> dict:
    records = list(read_completed(output_dir / "frames.jsonl").values())
    meta = {}
    meta_path = output_dir / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    statuses = Counter(item.get("status") for item in records)
    compute = []
    per_hypothesis = Counter()
    failure = Counter()
    support = []
    for item in records:
        timing = item.get("timing_s") or {}
        if "frame" in timing:
            compute.append(float(timing["frame"]))
        if item.get("status") in ("insufficient_geometry", "no_hypotheses", "error"):
            failure[item.get("status")] += 1
        if item.get("support_s_min_m") is not None and item.get("support_s_max_m") is not None:
            support.append((float(item["support_s_min_m"]), float(item["support_s_max_m"])))
        for hypothesis in item.get("hypotheses") or []:
            total = sum((hypothesis.get("candidate_counts") or {}).values())
            per_hypothesis[hypothesis.get("hypothesis_id")] += total
    compute_arr = np.asarray(compute, dtype=np.float64) if compute else np.zeros(0)
    evaluated = statuses.get("ok", 0)
    geometry_miss = statuses.get("insufficient_geometry", 0) + statuses.get("no_hypotheses", 0)
    denom = max(len(records), 1)
    return {
        "run": meta.get("run", output_dir.name),
        "algorithm_version": meta.get("algorithm_version"),
        "variant": meta.get("variant"),
        "topic": meta.get("topic"),
        "topic_reason": meta.get("topic_reason"),
        "coordinate_prior": meta.get("coordinate_prior"),
        "expected_messages": meta.get("expected_messages"),
        "processed_messages": len(records),
        "hardware": meta.get("hardware"),
        "status_counts": dict(statuses),
        "errors": int(statuses.get("error", 0)),
        "frames_without_usable_geometry": int(geometry_miss),
        "share_without_usable_geometry": geometry_miss / denom,
        "frames_ok": int(evaluated),
        "support_s_min_m": None if not support else min(item[0] for item in support),
        "support_s_max_m": None if not support else max(item[1] for item in support),
        "candidate_records_by_hypothesis_id": dict(per_hypothesis),
        "candidate_sum_is_not_an_obstacle_count": True,
        "compute_p50_s": None if compute_arr.size == 0 else float(np.percentile(compute_arr, 50)),
        "compute_p95_s": None if compute_arr.size == 0 else float(np.percentile(compute_arr, 95)),
        "compute_sum_s": float(compute_arr.sum()) if compute_arr.size else 0.0,
        "no_cross_frame_state": True,
    }


def select_extra_frames(records: list[dict]) -> dict:
    """Fixed after-the-fact picture rule. It does not change the detector."""
    best_count = None
    best_index = None
    failures = Counter()
    failure_first = {}
    for item in records:
        index = int(item["frame_index"])
        status = item.get("status")
        if status in ("insufficient_geometry", "no_hypotheses", "error"):
            failures[status] += 1
            failure_first.setdefault(status, index)
        peak = 0
        for hypothesis in item.get("hypotheses") or []:
            peak = max(peak, sum((hypothesis.get("candidate_counts") or {}).values()))
        if best_count is None or peak > best_count or (peak == best_count and index < best_index):
            best_count = peak
            best_index = index
    frequent_failure = None
    if failures:
        frequent_failure = sorted(failures, key=lambda key: (-failures[key], failure_first[key]))[0]
    return {
        "max_candidates_on_one_hypothesis": {"frame_index": best_index, "count": best_count},
        "most_common_geometry_failure": None
        if frequent_failure is None
        else {"status": frequent_failure, "count": int(failures[frequent_failure]), "first_frame_index": failure_first[frequent_failure]},
        "candidate_sum_was_not_used": True,
    }


def inventory(dataset: Path, config_dir: Path, output: Path) -> None:
    _corridor, _detector, _batch, _mount = load_runtime(config_dir)
    roles = load_run_roles((config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    proposal = []
    for child in sorted(dataset.iterdir()):
        if not child.is_dir() or not (child / "metadata.yaml").is_file():
            continue
        meta = read_metadata(child)
        role = roles.get(child.name, "unassigned")
        proposal.append(
            {
                "run": child.name,
                "proposed_role": role,
                "already_in_split": child.name in roles,
                "message_count": meta["message_count"],
                "topics": meta["topics"],
                "detector_started": False,
                "note": "Не запускать подбор параметров. Роль unassigned не обрабатывается.",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"inventory {len(proposal)} bags, detector not started", flush=True)


def _development_bags(dataset: Path, config_dir: Path) -> list[Path]:
    roles = load_run_roles((config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    return [dataset / name for name, role in roles.items() if role == "development"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Потоковый запуск однокадрового детектора")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--bag", type=Path)
    run.add_argument("--dataset", type=Path)
    run.add_argument("--development-set", action="store_true")
    run.add_argument("--config-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--topic", default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--message-index", type=int, action="append", default=None)
    run.add_argument("--from-ns", type=int, default=None)
    run.add_argument("--to-ns", type=int, default=None)
    run.add_argument("--pictures", action="store_true")
    run.add_argument("--pictures-only", action="store_true")
    inv = sub.add_parser("inventory")
    inv.add_argument("--dataset", type=Path, required=True)
    inv.add_argument("--config-dir", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    extra = sub.add_parser("select-pictures")
    extra.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        inventory(args.dataset, args.config_dir, args.output)
        return 0
    if args.command == "select-pictures":
        records = list(read_completed(args.output / "frames.jsonl").values())
        selection = select_extra_frames(records)
        (args.output / "picture_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    roles = load_run_roles((args.config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    if args.development_set:
        if args.dataset is None:
            raise SystemExit("нужен --dataset")
        bags = _development_bags(args.dataset, args.config_dir)
    elif args.bag is not None:
        bags = [args.bag]
    else:
        raise SystemExit("нужен --bag или --development-set")
    indices = None if not args.message_index else set(args.message_index)
    for bag in bags:
        target = args.output / bag.name if args.development_set else args.output
        run_bag(
            bag,
            target,
            args.config_dir,
            args.variant,
            roles,
            args.topic,
            args.max_frames,
            indices,
            args.from_ns,
            args.to_ns,
            args.pictures,
            pictures_only=args.pictures_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
