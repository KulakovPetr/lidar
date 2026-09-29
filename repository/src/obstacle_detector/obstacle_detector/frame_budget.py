"""Bound one frame by a killable worker process.

A Python thread that keeps computing after the deadline is not used.
The budget is an execution limit, not a detection threshold.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty

import numpy as np


class _PersistentWorker:
    """One process for many frames. A timeout kills it and the next frame starts a new one.

    `fork` avoids importing the stack again on every frame. `spawn` remains the
    fallback where fork is not available. Arguments are still sent once per frame;
    the interpreter is not.
    """

    def __init__(self) -> None:
        # spawn, not fork: the bag reader and ROS already have threads, and a
        # forked child deadlocks on locks those threads were holding.
        self._method = "spawn"
        self._context = None
        self._process = None
        self._inbox = None
        self._outbox = None
        self._open()

    def _open(self) -> None:
        self._context = mp.get_context(self._method)
        self._inbox = self._context.Queue()
        self._outbox = self._context.Queue(maxsize=1)
        self._process = self._context.Process(target=_serve, args=(self._inbox, self._outbox), daemon=True)
        self._process.start()

    def _close(self) -> bool:
        process = self._process
        alive = bool(process.is_alive()) if process is not None else False
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        alive_after = bool(process.is_alive()) if process is not None else False
        self._process = None
        self._inbox = None
        self._outbox = None
        return alive_after

    def call(self, target, args: tuple, budget_s: float, progress_path: str | None) -> dict:
        if self._process is None or not self._process.is_alive():
            self._open()
        started = time.perf_counter()
        args = _compact_group_indices(args)
        prepare_s = time.perf_counter() - started
        put_started = time.perf_counter()
        self._inbox.put((target, args, progress_path))
        put_s = time.perf_counter() - put_started
        try:
            kind, payload = self._outbox.get(timeout=float(budget_s))
        except Empty:
            kind, payload = None, None
        elapsed = time.perf_counter() - started
        if kind is None:
            alive_after = self._close()
            self._open()
            progress = None
            if progress_path and Path(progress_path).is_file():
                try:
                    progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    progress = {"unreadable": True}
            return {
                "status": "processing_timeout",
                "elapsed_s": elapsed,
                "budget_s": float(budget_s),
                "budget_is_not_a_detection_threshold": True,
                "worker_alive_after_stop": alive_after,
                "worker_pid": None,
                "cluster_progress": progress,
                "absence_of_candidates_is_not_clear": True,
                "empty_result_is_not_clear": True,
            }
        if kind == "error":
            return {
                "status": "processing_error",
                "elapsed_s": elapsed,
                "error": payload,
                "worker_alive_after_stop": self._process.is_alive(),
                "absence_of_candidates_is_not_clear": True,
            }
        payload = dict(payload)
        payload["worker_elapsed_s"] = elapsed
        payload["worker_alive_after_stop"] = False
        payload.setdefault("timing_s", {})
        if isinstance(payload["timing_s"], dict):
            payload["timing_s"]["ipc_prepare_s"] = prepare_s
            payload["timing_s"]["ipc_put_s"] = put_s
        return payload


_WORKER: _PersistentWorker | None = None


def ensure_worker() -> _PersistentWorker:
    global _WORKER
    if _WORKER is None:
        _WORKER = _PersistentWorker()
    return _WORKER


def _serve(inbox, outbox) -> None:
    import os

    limit_library_threads()
    while True:
        item = inbox.get()
        if item is None:
            return
        target, args, progress_path = item
        if progress_path:
            os.environ["OBSTACLE_CLUSTER_PROGRESS"] = progress_path
        try:
            outbox.put(("ok", target(*args)))
        except Exception:
            outbox.put(("error", traceback.format_exc()))


def _compact_group_indices(args: tuple) -> tuple:
    """Send the first original row of each exact-XYZ group as one numeric array.

    The detector reads only that first row. Pickling an object array of member
    lists repeats the cloud on every frame and does not change the decision.
    """
    if not args or not isinstance(args[0], dict) or "groups" not in args[0]:
        return args
    payload = dict(args[0])
    groups = payload["groups"]
    if not isinstance(groups, dict) or "indices" not in groups:
        return args
    indices = groups["indices"]
    if not isinstance(indices, np.ndarray) or indices.dtype != object:
        return args
    groups = dict(groups)
    if "first_index" in groups:
        first = np.asarray(groups["first_index"], dtype=np.int64).reshape(-1, 1)
    else:
        count = int(indices.shape[0]) if indices.ndim else 0
        first = np.empty((count, 1), dtype=np.int64)
        for row, members in enumerate(indices):
            first[row, 0] = int(np.asarray(members).reshape(-1)[0])
    groups["indices"] = first
    payload["groups"] = groups
    return (payload,) + tuple(args[1:])


def call_bounded(target, args: tuple, budget_s: float, progress_path: str | None = None) -> dict:
    if budget_s <= 0:
        raise ValueError("frame budget must be positive")
    return ensure_worker().call(target, args, budget_s, progress_path)


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_library_threads() -> None:
    """One thread per worker. workers times this stays within the machine."""
    for name in _THREAD_VARS:
        os.environ[name] = "1"


class CloudBuffers:
    """Fixed shared-memory slots. A slot stays busy while a worker still reads it."""

    def __init__(self, count: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self.slots = []
        for _ in range(int(count)):
            shm = shared_memory.SharedMemory(create=True, size=self.capacity)
            self.slots.append({"name": shm.name, "shm": shm, "busy": False, "extra": False})
        self.extra = {}

    def write(self, payload) -> tuple[str, int]:
        data = memoryview(payload)
        length = int(data.nbytes)
        with self._lock:
            return self._write_locked(data, length)

    def _write_locked(self, data, length: int) -> tuple[str, int]:
        for slot in self.slots:
            if not slot["busy"] and length <= self.capacity:
                slot["shm"].buf[:length] = data
                slot["busy"] = True
                return slot["name"], length
        shm = shared_memory.SharedMemory(create=True, size=length)
        shm.buf[:length] = data
        self.extra[shm.name] = shm
        return shm.name, length

    def release(self, name: str) -> None:
        with self._lock:
            self._release_locked(name)

    def _release_locked(self, name: str) -> None:
        for slot in self.slots:
            if slot["name"] == name:
                slot["busy"] = False
                return
        shm = self.extra.pop(name, None)
        if shm is not None:
            shm.close()
            shm.unlink()

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        for slot in self.slots:
            try:
                slot["shm"].close()
                slot["shm"].unlink()
            except FileNotFoundError:
                pass
        self.slots.clear()
        for shm in self.extra.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self.extra.clear()


class WorkerPool:
    """Persistent processes. Each reply is read from the worker that received that job."""

    def __init__(self, workers: int) -> None:
        if int(workers) not in (1, 2, 4, 6):
            raise ValueError("workers must be 1, 2, 4 or 6")
        limit_library_threads()
        self.workers = int(workers)
        self._context = mp.get_context("spawn")
        self.slots = [self._open() for _ in range(self.workers)]

    def _open(self) -> dict:
        inbox = self._context.Queue()
        outbox = self._context.Queue(maxsize=1)
        process = self._context.Process(target=_serve_pool, args=(inbox, outbox), daemon=True)
        process.start()
        return {"process": process, "inbox": inbox, "outbox": outbox, "job_id": None}

    def has_free(self) -> bool:
        return any(slot["job_id"] is None for slot in self.slots)

    def submit(self, job_id: int, target, args: tuple, progress_path: str | None) -> None:
        for slot in self.slots:
            if slot["job_id"] is not None:
                continue
            if slot["process"] is None or not slot["process"].is_alive():
                replacement = self._open()
                slot.update(replacement)
            slot["job_id"] = int(job_id)
            slot["inbox"].put((int(job_id), target, args, progress_path))
            return
        raise RuntimeError("no free worker")

    def poll(self, now: float, deadlines: dict[int, float]) -> list[dict]:
        events = []
        for index, slot in enumerate(self.slots):
            job_id = slot["job_id"]
            if job_id is None:
                continue
            try:
                kind, payload = slot["outbox"].get_nowait()
            except Empty:
                if now > float(deadlines.get(int(job_id), now + 1.0)):
                    events.append(self._stop_slot(index, int(job_id)))
                continue
            slot["job_id"] = None
            if kind == "error":
                events.append(
                    {
                        "status": "processing_error",
                        "job_id": int(job_id),
                        "error": payload,
                        "absence_of_candidates_is_not_clear": True,
                    }
                )
                continue
            payload = dict(payload)
            payload["job_id"] = int(job_id)
            events.append(payload)
        return events

    def _stop_slot(self, index: int, job_id: int) -> dict:
        slot = self.slots[index]
        process = slot["process"]
        alive_after = False
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            alive_after = bool(process.is_alive())
        self.slots[index] = self._open()
        return {
            "status": "processing_timeout",
            "job_id": int(job_id),
            "budget_is_not_a_detection_threshold": True,
            "worker_alive_after_stop": alive_after,
            "worker_replaced": True,
            "absence_of_candidates_is_not_clear": True,
            "empty_result_is_not_clear": True,
        }

    def shutdown(self) -> None:
        for slot in self.slots:
            process = slot["process"]
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
        self.slots.clear()


def _serve_pool(inbox, outbox) -> None:
    limit_library_threads()
    while True:
        item = inbox.get()
        if item is None:
            return
        job_id, target, args, progress_path = item
        if progress_path:
            os.environ["OBSTACLE_CLUSTER_PROGRESS"] = progress_path
        try:
            payload = target(*args)
            if isinstance(payload, dict):
                payload["job_id"] = int(job_id)
            outbox.put(("ok", payload))
        except Exception:
            outbox.put(("error", traceback.format_exc()))


def sleep_for_budget_test(seconds: float) -> dict:
    time.sleep(float(seconds))
    return {"status": "ok", "slept_s": float(seconds)}
