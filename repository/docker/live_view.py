"""ROS web view. It draws the display sample, not a saved picture and not the detector input."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, String

PAGE = r"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>LiDAR live</title>
<style>
body { margin: 0; background: #1C1D22; color: #FFF7FB; font: 16px/1.35 sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 14px 18px; align-items: baseline; padding: 10px 16px; background: #310F53; }
h1 { font-size: 18px; margin: 0; }
.range { color: #FF0053; font-size: 28px; font-weight: 700; }
.banner { display: none; margin: 8px 16px; padding: 10px 14px; background: #FFF7FB; color: #1C1D22; border-radius: 8px; }
.banner.show { display: block; }
.stage { padding: 8px 16px 0; }
canvas { width: 100%; height: 68vh; background: #111; border-radius: 8px; }
.row { display: flex; gap: 8px; padding: 8px 16px 16px; }
button { background: #FFF7FB; color: #1C1D22; border: 0; border-radius: 6px; padding: 8px 12px; }
body.record header, body.record .row, body.record details { display: none; }
body.record .strip { display: flex; }
body.record canvas { height: 86vh; }
.strip { display: none; gap: 18px; align-items: baseline; padding: 8px 16px; background: #310F53; }
.note { color: #FF0053; }
</style></head>
<body>
<div class="strip">
  <b id="strip-mode">—</b>
  <span>результаты <b id="strip-rate">—</b></span>
  <span class="range" id="strip-range">—</span>
</div>
<header>
  <h1>Живой просмотр ROS</h1>
  <div>режим <b id="mode">—</b></div>
  <div>кадр <b id="seq">—</b></div>
  <div>время bag <b id="bag">—</b></div>
  <div>обработка <b id="work">—</b></div>
  <div>дальность <b class="range" id="range">—</b></div>
  <div>статус <b id="status">—</b></div>
</header>
<div id="banner" class="banner"></div>
<div class="stage"><canvas id="view"></canvas></div>
<div class="row">
  <div>перерисовка страницы <b id="browser">—</b></div>
  <div>вход <b id="input">—</b></div>
  <div>результаты <b id="result">—</b></div>
  <div>частоты <b id="rates">—</b></div>
  <div>среднее с открытия <b id="since">—</b></div>
  <div>пропуски <b id="drops">0</b></div>
  <div>ошибки <b id="errors">0</b></div>
  <div>таймаут <b id="timeouts">0</b></div>
  <button id="hold" type="button">Пауза</button>
  <button id="record" type="button">Экран записи</button>
</div>
<details><summary>Профиль и допущения</summary><pre id="tech"></pre></details>
<script>
const canvas = document.getElementById("view");
let paints = 0, browserHz = 0, holding = false;
function frame() { paints += 1; requestAnimationFrame(frame); }
requestAnimationFrame(frame);
setInterval(() => { browserHz = paints; paints = 0; }, 1000);
function fit() {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, r.width * devicePixelRatio);
  canvas.height = Math.max(1, r.height * devicePixelRatio);
}
function draw(points) {
  fit();
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!points || !points.length) {
    g.fillStyle = "#888"; g.font = "24px sans-serif"; g.fillText("нет кадра", 24, 48); return;
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, -p[1]); maxY = Math.max(maxY, -p[1]);
  }
  const span = Math.max(maxX - minX, maxY - minY, 1);
  const scale = Math.min(canvas.width, canvas.height) / span * 0.86;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  function xy(p) { return [canvas.width / 2 + (p[0] - cx) * scale, canvas.height / 2 - ((-p[1]) - cy) * scale]; }
  for (const p of points) {
    if (p[3] > 0.5) continue;
    const [x, y] = xy(p);
    g.fillStyle = "#9ad";
    g.fillRect(x, y, 2, 2);
  }
  for (const p of points) {
    if (p[3] <= 0.5) continue;
    const [x, y] = xy(p);
    g.fillStyle = "#FF0053";
    g.fillRect(x - 2, y - 2, 5, 5);
  }
}
function text(id, value) { document.getElementById(id).textContent = value; }
async function tick() {
  const state = await (await fetch("/state.json")).json();
  const range = state.range_m == null ? "—" : state.range_m + " м";
  text("mode", state.mode_label || "—");
  text("seq", state.sequence == null ? "—" : (state.sequence + " / " + (state.received ?? "—")));
  text("bag", state.bag_time_s == null ? "—" : state.bag_time_s + " с");
  text("work", state.processing_s == null ? "—" : state.processing_s + " с");
  text("range", range);
  text("status", state.status_label || "—");
  text("browser", browserHz + " /с");
  text("input", (state.input_hz ?? "—") + " сообщ/с");
  text("result", (state.compact_hz ?? state.result_hz ?? "—") + " компакт/с");
  text("rates", state.rates || "—");
  text("since", (state.result_hz_since_open ?? "—") + " результат/с");
  text("drops", state.drops ?? 0);
  text("errors", state.errors ?? 0);
  text("timeouts", state.timeouts ?? 0);
  text("strip-mode", state.mode_label || "—");
  text("strip-rate", (state.result_hz ?? "—") + " результат/с");
  text("strip-range", range);
  document.getElementById("tech").textContent = state.technical || "";
  const banner = document.getElementById("banner");
  if (state.finished) {
    banner.className = "banner show";
    banner.textContent = "Прогон завершён. " + (state.finished_text || "");
  } else if (state.timeout_frame) {
    banner.className = "banner show";
    banner.textContent = "Таймаут этого кадра. Чужой результат не показан.";
  } else if (state.newer_input) {
    banner.className = "banner show";
    banner.textContent = "Это результат более раннего кадра. Новое облако уже во входе и ещё не обработано.";
  } else {
    banner.className = "banner";
    banner.textContent = "";
  }
  holding = !!state.hold;
  const pause = document.getElementById("hold");
  pause.textContent = holding ? "Продолжить" : "Пауза";
  pause.disabled = !state.can_pause;
  draw(state.points || []);
}
document.getElementById("hold").onclick = async () => {
  await fetch("/control", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({hold: !holding})});
};
document.getElementById("record").onclick = () => document.body.classList.toggle("record");
if (location.search.indexOf("record=1") >= 0) document.body.classList.add("record");
setInterval(() => tick().catch(() => {}), 200);
tick();
</script>
</body></html>
"""


def _window_rate(samples: deque, now: float, window_s: float) -> float | None:
    if len(samples) < 2:
        return None
    while samples and now - samples[0][0] > window_s:
        samples.popleft()
    if len(samples) < 2:
        return None
    dt = samples[-1][0] - samples[0][0]
    if dt <= 0:
        return None
    return (samples[-1][1] - samples[0][1]) / dt


class Live(Node):
    def __init__(self) -> None:
        super().__init__("live_view")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PointCloud2, "/display_cloud", self._on_cloud, qos)
        self.create_subscription(String, "/detection", self._on_detection, qos)
        self.create_subscription(String, "/diagnostics", self._on_diag, qos)
        self.create_subscription(String, "/run_status", self._on_status, qos)
        self._hold_pub = self.create_publisher(Bool, "/playback_hold", qos)
        self._lock = threading.Lock()
        self._cloud = None
        self._cloud_seq = None
        self._card = None
        self._shown = None
        self._diag = {}
        self._status = {}
        self._hold = False
        self._t0 = time.monotonic()
        self._input_samples = deque(maxlen=40)
        self._result_samples = deque(maxlen=40)
        self._results = 0
        self._first_bag_ns = None
        self._mode = os.environ.get("MODE", "stream")
        self._rate = os.environ.get("RATE", "1.0")
        self.create_timer(1.0, self._republish_hold)
        self.create_timer(1.0, self._pulse)

    def _republish_hold(self) -> None:
        msg = Bool()
        msg.data = self._hold
        self._hold_pub.publish(msg)

    def set_hold(self, hold: bool) -> None:
        self._hold = bool(hold)
        self._republish_hold()

    def _pulse(self) -> None:
        path = os.environ.get("LOG_DIR", "/output")
        try:
            with self._lock:
                shown = self._shown or {}
                text = (
                    f"mode={self._mode} seq={shown.get('sequence')} status={shown.get('status')} "
                    f"range={shown.get('range_from_lidar_m')} received={self._diag.get('received')}\n"
                )
            with open(os.path.join(path, "pulse.txt"), "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            return

    def _on_cloud(self, msg: PointCloud2) -> None:
        frame_id = str(msg.header.frame_id)
        if "#" not in frame_id:
            return
        try:
            sequence = int(frame_id.rsplit("#", 1)[1])
        except ValueError:
            return
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label", "<f4")])
        count = int(msg.width) * int(msg.height)
        raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=count)
        points = np.column_stack((raw["x"], raw["y"], raw["z"], raw["label"])).round(2)
        with self._lock:
            self._cloud = points
            self._cloud_seq = sequence
            self._match()

    def _on_detection(self, msg: String) -> None:
        try:
            card = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._card = card
            self._match()

    def _match(self) -> None:
        card = self._card
        if card is None or self._cloud is None or self._cloud_seq is None:
            return
        if int(card.get("sequence", -1)) != int(self._cloud_seq):
            return
        labels = self._cloud[:, 3]
        highlight = int(np.count_nonzero(labels > 0.5))
        expected = int(card.get("highlight_count") or 0)
        bag_ns = card.get("bag_time_ns")
        if bag_ns is not None and self._first_bag_ns is None:
            self._first_bag_ns = int(bag_ns)
        self._shown = {
            "card": card,
            "points": self._cloud,
            "highlight_drawn": highlight,
            "highlight_complete": highlight == expected,
            "sequence": int(card.get("sequence")),
            "status": card.get("status"),
            "range_from_lidar_m": card.get("range_from_lidar_m"),
        }
        self._results += 1
        now = time.monotonic()
        self._result_samples.append((now, self._results))

    def _on_diag(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = time.monotonic()
        with self._lock:
            self._diag = payload
            self._input_samples.append((now, int(payload.get("received") or 0)))

    def _on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._status = payload

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            shown = self._shown or {}
            card = shown.get("card") or {}
            points = shown.get("points")
            received = int(self._diag.get("received") or 0)
            elapsed = max(now - self._t0, 1e-3)
            bag_ns = card.get("bag_time_ns")
            bag_s = None
            if bag_ns is not None and self._first_bag_ns is not None:
                bag_s = round((int(bag_ns) - self._first_bag_ns) / 1e9, 3)
            profile = card.get("profile") or {}
            status = card.get("status")
            if self._mode == "sequential":
                mode_label = "последовательный плеер, не ros2 bag play"
            else:
                mode_label = f"ros2 bag play {self._rate}×, не частота обработки"
            finished = self._status.get("state") == "finished"
            counts = self._status.get("counts") or {}
            finished_text = ""
            if finished and counts:
                finished_text = (
                    f"сообщений {self._status.get('messages_played')}, "
                    f"ok {counts.get('ok')}, таймаут {counts.get('timeout')}, "
                    f"ошибки {counts.get('error')}, ожидание без ответа {counts.get('player_wait_expired')}"
                )
            elif finished:
                finished_text = (
                    f"вход {self._diag.get('received')}, обработано {self._diag.get('completed_ok')}, "
                    f"пропуски {self._diag.get('dropped')}, ошибки {self._diag.get('processing_error')}, "
                    f"таймаут {self._diag.get('processing_timeout')}"
                )
            technical = {
                "profile": profile,
                "assumptions": [
                    "ориентация и нижняя граница не подтверждены",
                    "горизонт поиска не подтверждённый путь",
                    "отсутствие кандидатов не означает свободный путь",
                    "прореживание только для рисунка",
                ],
                "decision": card.get("decision"),
                "evidence": card.get("evidence"),
                "corridor_segment": card.get("corridor_segment"),
                "highlight_drawn": shown.get("highlight_drawn"),
                "highlight_complete": shown.get("highlight_complete"),
                "result_is_answer_to_latest_input": card.get("result_is_answer_to_latest_input"),
                "processing_s": card.get("result_elapsed_s"),
                "timing_s": card.get("timing_s"),
                "display_cloud_is_not_the_detector_input": True,
            }
            return {
                "mode_label": mode_label,
                "sequence": shown.get("sequence"),
                "received": received,
                "bag_time_s": bag_s,
                "processing_s": None if card.get("result_elapsed_s") is None else round(float(card["result_elapsed_s"]), 3),
                "range_m": None if card.get("range_from_lidar_m") is None else round(float(card["range_from_lidar_m"]), 3),
                "status_label": card.get("display_label") or status or "—",
                "status": status,
                "input_hz": self._diag.get("input_hz"),
                "completed_hz": self._diag.get("completed_hz"),
                "compact_hz": self._diag.get("compact_hz"),
                "viz_prepared_hz": self._diag.get("viz_prepared_hz"),
                "viz_published_hz": self._diag.get("viz_published_hz"),
                "shown_age_s": self._diag.get("shown_age_s"),
                "result_hz": _round(_window_rate(self._result_samples, now, 5.0)),
                "rates": (
                    f"вход {self._diag.get('input_hz')} /с, "
                    f"расчёт {self._diag.get('completed_hz')} /с, "
                    f"новые кадры {self._diag.get('viz_prepared_hz')} /с, "
                    f"предел картинки {self._diag.get('visualization_rate_limit_hz')} Гц, "
                    f"возраст {self._diag.get('shown_age_s')} с, "
                    f"пропуски {self._diag.get('dropped')}"
                ),
                "result_hz_since_open": round(self._results / elapsed, 3),
                "drops": int(self._diag.get("dropped") or 0),
                "errors": int(self._diag.get("processing_error") or 0),
                "timeouts": int(self._diag.get("processing_timeout") or 0),
                "timeout_frame": status == "processing_timeout",
                "newer_input": bool(card.get("newer_input_exists")),
                "finished": finished,
                "finished_text": finished_text,
                "hold": self._hold,
                "can_pause": self._mode == "sequential",
                "points": [] if points is None else points.tolist(),
                "identity": (
                    f"{self._diag.get('corridor_source') or '—'} "
                    f"{self._diag.get('profile_name') or ''} "
                    f"workers={self._diag.get('workers')} backend={self._diag.get('compute_backend')} "
                    f"algo={self._diag.get('algorithm_version')} cfg={self._diag.get('config_version')}"
                ),
                "technical": json.dumps(technical, ensure_ascii=False),
            }


def _round(value):
    if value is None:
        return None
    return round(float(value), 3)


class Handler(BaseHTTPRequestHandler):
    node: Live

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/state.json"):
            body = json.dumps(self.node.snapshot(), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        if not self.path.startswith("/control"):
            self._send(404, b"", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        self.node.set_hold(bool(payload.get("hold")))
        self._send(200, b'{"ok":true}', "application/json")


def main() -> None:
    rclpy.init()
    node = Live()
    Handler.node = node
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
