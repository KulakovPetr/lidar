"""Streaming bag access. The whole bag is not loaded into memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from obstacle_detector.pointcloud import FieldSpec, decode_cloud


def decode_message(msg):
    """Same field decode as the stage-4A reader. XYZ stay float32."""
    fields = [FieldSpec(field.name, field.offset, field.datatype, field.count) for field in msg.fields]
    raw = np.asarray(msg.data, dtype=np.uint8)
    decoded, layout = decode_cloud(
        raw, msg.height, msg.width, msg.point_step, msg.row_step, bool(msg.is_bigendian), fields
    )
    def optional(name, dtype=None):
        if name not in decoded:
            return None
        values = np.asarray(decoded[name])
        return values if dtype is None else values.astype(dtype, copy=False)

    return {
        "layout": layout,
        "x": np.asarray(decoded["x"], dtype=np.float32),
        "y": np.asarray(decoded["y"], dtype=np.float32),
        "z": np.asarray(decoded["z"], dtype=np.float32),
        "intensity": np.asarray(decoded["intensity"], dtype=np.float32) if "intensity" in decoded else None,
        "ring": optional("ring"),
        "timestamp": optional("timestamp", np.float64),
        "absent_fields": [name for name in ("intensity", "ring", "timestamp") if name not in decoded],
        "width": int(msg.width),
        "height": int(msg.height),
        "frame_id": str(msg.header.frame_id),
        "stamp_ns": int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
    }


def decode_recorded(raw: np.ndarray, spec: dict) -> dict:
    """Decode a PointCloud2 payload that was copied out of the ROS callback."""
    fields = [FieldSpec(name, offset, datatype, count) for name, offset, datatype, count in spec["fields"]]
    decoded, layout = decode_cloud(
        raw,
        int(spec["height"]),
        int(spec["width"]),
        int(spec["point_step"]),
        int(spec["row_step"]),
        bool(spec["is_bigendian"]),
        fields,
    )

    def optional(name, dtype=None):
        if name not in decoded:
            return None
        values = np.asarray(decoded[name])
        return values if dtype is None else values.astype(dtype, copy=False)

    return {
        "layout": layout,
        "x": np.asarray(decoded["x"], dtype=np.float32),
        "y": np.asarray(decoded["y"], dtype=np.float32),
        "z": np.asarray(decoded["z"], dtype=np.float32),
        "intensity": np.asarray(decoded["intensity"], dtype=np.float32) if "intensity" in decoded else None,
        "ring": optional("ring"),
        "timestamp": optional("timestamp", np.float64),
        "absent_fields": [name for name in ("intensity", "ring", "timestamp") if name not in decoded],
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "frame_id": str(spec["frame_id"]),
        "stamp_ns": int(spec["stamp_ns"]),
    }


def open_reader(bag_dir: Path):
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def choose_pointcloud_topic(topics, known_topics, explicit: str | None = None) -> tuple[str, str]:
    """Pick one PointCloud2 topic. Several topics are never merged."""
    names = [topic.name for topic in topics if topic.type == "sensor_msgs/msg/PointCloud2"]
    if explicit:
        if explicit not in names:
            raise SystemExit(f"топик {explicit} не является PointCloud2 в этом bag: {names}")
        return explicit, "явный выбор"
    if len(names) == 1:
        return names[0], "единственный топик PointCloud2"
    known = [name for name in names if name in set(known_topics)]
    if len(known) == 1:
        return known[0], "единственный известный топик из нескольких PointCloud2"
    if not names:
        raise SystemExit("в bag нет топика PointCloud2")
    raise SystemExit("несколько топиков PointCloud2, нужен --topic; они не объединяются: " + ", ".join(names))


def read_metadata(bag_dir: Path) -> dict:
    path = bag_dir / "metadata.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    info = payload.get("rosbag2_bagfile_information", payload)
    topics = []
    for item in info.get("topics_with_message_count") or []:
        meta = item.get("topic_metadata") or {}
        topics.append(
            {
                "name": meta.get("name"),
                "type": meta.get("type"),
                "message_count": int(item.get("message_count") or 0),
            }
        )
    return {
        "path": str(bag_dir),
        "message_count": int(info.get("message_count") or 0),
        "duration_ns": (info.get("duration") or {}).get("nanoseconds"),
        "starting_time_ns": (info.get("starting_time") or {}).get("nanoseconds_since_epoch"),
        "topics": topics,
    }
