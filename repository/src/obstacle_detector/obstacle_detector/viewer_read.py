"""Read one stored cloud for the viewer. This is not the detector."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


def _topic_ids(connection, topic: str) -> list[int]:
    rows = connection.execute(
        "SELECT m.id FROM messages m JOIN topics t ON m.topic_id = t.id WHERE t.name = ? ORDER BY m.id",
        (topic,),
    ).fetchall()
    return [int(row[0]) for row in rows]


def read_cloud(bag_dir: Path, message_index: int, topic: str, expect_stamp: int | None, highlight: list[int], background_limit: int, points_cache: Path | None = None) -> dict:
    databases = sorted(bag_dir.glob("*.db3"))
    if len(databases) != 1:
        raise SystemExit(f"ожидался один файл .db3, найдено {len(databases)}")
    connection = sqlite3.connect(f"file:{databases[0].as_posix()}?mode=ro", uri=True)
    ids = _topic_ids(connection, topic)
    if message_index < 0 or message_index >= len(ids):
        raise SystemExit(f"индекс {message_index} вне диапазона топика {topic}: {len(ids)}")
    blob = connection.execute("SELECT data FROM messages WHERE id = ?", (ids[message_index],)).fetchone()
    if blob is None:
        raise SystemExit("сообщение не найдено")
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    from obstacle_detector.bag_io import decode_message

    decoded = decode_message(deserialize_message(blob[0], PointCloud2))
    stamp = int(decoded["stamp_ns"])
    if expect_stamp is not None and stamp != int(expect_stamp):
        raise SystemExit(f"метка облака {stamp} не совпала с результатом {expect_stamp}")
    total = int(decoded["x"].shape[0])
    step = max(1, int(np.ceil(total / max(1, background_limit))))
    background_rows = np.arange(0, total, step, dtype=np.int64)
    chosen = np.asarray(highlight, dtype=np.int64)
    chosen = chosen[(chosen >= 0) & (chosen < total)]
    if points_cache is not None:
        packed = np.column_stack([decoded["x"], decoded["y"], decoded["z"]]).astype(np.float32)
        points_cache.write_bytes(packed.tobytes())
    def pack(rows):
        if rows.size == 0:
            return []
        return np.column_stack([decoded["x"][rows], decoded["y"][rows], decoded["z"][rows]]).astype(np.float32).reshape(-1).tolist()
    return {
        "message_index": message_index,
        "topic": topic,
        "header_stamp_ns": stamp,
        "header_frame_id": decoded["frame_id"],
        "total_points": total,
        "background_stride": step,
        "background_xyz": pack(background_rows),
        "highlight_xyz": pack(chosen),
        "highlight_count": int(chosen.size),
        "source_indices_are_original_cloud_rows": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Одно облако для просмотра")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--expect-stamp", type=int, default=None)
    parser.add_argument("--highlight-file", type=Path, default=None)
    parser.add_argument("--background-limit", type=int, default=20000)
    parser.add_argument("--points-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    highlight = []
    if args.highlight_file is not None:
        highlight = json.loads(args.highlight_file.read_text(encoding="utf-8"))
    payload = read_cloud(
        args.bag, args.index, args.topic, args.expect_stamp, highlight, args.background_limit, args.points_cache
    )
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"points": payload["total_points"], "highlight": payload["highlight_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
