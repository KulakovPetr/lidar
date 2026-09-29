"""Sequential bag reader. This is not `ros2 bag play`.

The next cloud is published only after the detector returns a result for
this run_id and sequence: success, diagnostic refusal, error, or timeout.
A DDS acknowledgment of the cloud is not that result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String

from obstacle_detector.bag_io import open_reader
from obstacle_detector.offline_run import select_topic


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class SequentialPlayer(Node):
    def __init__(self, topic: str, run_id: str, budget_s: float) -> None:
        super().__init__("sequential_player")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._topic = topic
        self._run_id = run_id
        self._budget_s = float(budget_s)
        self._cloud_pub = self.create_publisher(PointCloud2, topic, qos)
        self._token_pub = self.create_publisher(String, "/frame_token", qos)
        self._status_pub = self.create_publisher(String, "/run_status", qos)
        self.create_subscription(String, "/detection", self._on_detection, qos)
        self.create_subscription(Bool, "/playback_hold", self._on_hold, qos)
        self._answer = None
        self._hold = False

    def _on_detection(self, msg: String) -> None:
        try:
            self._answer = json.loads(msg.data)
        except json.JSONDecodeError:
            self._answer = None

    def _on_hold(self, msg: Bool) -> None:
        self._hold = bool(msg.data)

    def wait_for_detector(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.count_subscribers(self._topic) >= 1 and self.count_subscribers("/frame_token") >= 1:
                return True
        return False

    def _wait_if_paused(self) -> None:
        while self._hold:
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_and_wait(self, cloud: PointCloud2, sequence: int, bag_time_ns: int) -> dict:
        token = {
            "run_id": self._run_id,
            "sequence": int(sequence),
            "bag_time_ns": int(bag_time_ns),
            "stamp_ns": _stamp_ns(cloud.header),
            "frame_id": str(cloud.header.frame_id),
            "player": "sequential_bag_reader_not_ros2_bag_play",
            "dds_ack_is_not_compute_completion": True,
        }
        self._answer = None
        self._token_pub.publish(String(data=json.dumps(token)))
        self._cloud_pub.publish(cloud)
        deadline = time.monotonic() + self._budget_s + 15.0
        started = time.perf_counter()
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            answer = self._answer
            if not isinstance(answer, dict):
                continue
            if answer.get("run_id") != self._run_id:
                continue
            if int(answer.get("sequence", -1)) != int(sequence):
                continue
            answer = dict(answer)
            answer["player_wait_s"] = time.perf_counter() - started
            return answer
        return {
            "run_id": self._run_id,
            "sequence": int(sequence),
            "status": "player_wait_expired",
            "decision": "player_wait_expired",
            "player_wait_s": time.perf_counter() - started,
            "dds_ack_is_not_compute_completion": True,
        }

    def publish_status(self, payload: dict) -> None:
        self._status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rclpy.spin_once(self, timeout_sec=0.1)


def _run_id(bag: Path) -> str:
    digest = hashlib.sha256(f"{bag}:{uuid.uuid4()}".encode()).hexdigest()
    return digest[:16]


def play(bag: Path, topic: str | None, output: Path, budget_s: float, start: int, count: int | None) -> dict:
    reader = open_reader(bag)
    chosen, reason = select_topic(list(reader.get_all_topics_and_types()), topic)
    run_id = _run_id(bag)
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = SequentialPlayer(chosen, run_id, budget_s)
    times = output / "playback_times.jsonl"
    counts = {"ok": 0, "timeout": 0, "error": 0, "player_wait_expired": 0, "repeated_stamps": 0}
    seen = 0
    played = 0
    last_stamp = None
    started = time.perf_counter()
    summary = {
        "player": "sequential_bag_reader_not_ros2_bag_play",
        "ros2_bag_play": False,
        "dds_ack_is_not_compute_completion": True,
        "run_id": run_id,
        "topic": chosen,
        "topic_reason": reason,
        "start": int(start),
        "count": count,
        "state": "running",
    }
    try:
        if not node.wait_for_detector():
            summary["state"] = "detector_not_ready"
            summary["elapsed_s"] = time.perf_counter() - started
            (output / "playback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            node.publish_status(summary)
            return summary
        print("SEQUENTIAL_PLAYER_READY", flush=True)
        print("THIS_IS_NOT_ROS2_BAG_PLAY", flush=True)
        with times.open("w", encoding="utf-8") as handle:
            while reader.has_next():
                read_started = time.perf_counter()
                topic_name, payload, record_ns = reader.read_next()
                bag_read_s = time.perf_counter() - read_started
                if topic_name != chosen:
                    continue
                if count is not None and seen >= start + count:
                    break
                index = seen
                seen += 1
                if index < start:
                    continue
                node._wait_if_paused()
                decode_started = time.perf_counter()
                cloud = deserialize_message(payload, PointCloud2)
                deserialize_s = time.perf_counter() - decode_started
                stamp = _stamp_ns(cloud.header)
                if last_stamp is not None and stamp == last_stamp:
                    counts["repeated_stamps"] += 1
                last_stamp = stamp
                try:
                    answer = node.publish_and_wait(cloud, index, int(record_ns))
                except Exception as exc:  # noqa: BLE001
                    answer = {
                        "run_id": run_id,
                        "sequence": index,
                        "status": "player_error",
                        "decision": "player_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                status = str(answer.get("status") or "")
                if status == "ok":
                    counts["ok"] += 1
                elif status == "processing_timeout":
                    counts["timeout"] += 1
                elif status == "player_wait_expired":
                    counts["player_wait_expired"] += 1
                else:
                    counts["error"] += 1
                played += 1
                handle.write(json.dumps({
                    "sequence": index,
                    "stamp_ns": stamp,
                    "frame_id": str(cloud.header.frame_id),
                    "status": status,
                    "decision": answer.get("decision"),
                    "player_wait_s": answer.get("player_wait_s"),
                    "result_elapsed_s": answer.get("result_elapsed_s"),
                    "bag_read_s": bag_read_s,
                    "deserialize_s": deserialize_s,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"sequence": index, "status": status, "wait_s": answer.get("player_wait_s")}, ensure_ascii=False), flush=True)
        summary["state"] = "finished"
    finally:
        summary["messages_seen"] = seen
        summary["messages_played"] = played
        summary["counts"] = counts
        summary["elapsed_s"] = time.perf_counter() - started
        summary["repeated_stamps_do_not_replace_sequence"] = True
        (output / "playback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            node.publish_status(summary)
            time.sleep(0.3)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Последовательный прогон bag. Это не ros2 bag play.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args(argv)
    if args.budget <= 0:
        raise SystemExit("--budget должен быть положительным")
    if args.start < 0 or (args.count is not None and args.count < 0):
        raise SystemExit("start и count должны быть неотрицательными")
    summary = play(args.bag, args.topic, args.output, args.budget, args.start, args.count)
    print(json.dumps({"state": summary["state"], "played": summary.get("messages_played"), "counts": summary.get("counts")}, ensure_ascii=False), flush=True)
    if summary.get("state") != "finished":
        return 1
    if int((summary.get("counts") or {}).get("player_wait_expired") or 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
