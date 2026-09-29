"""Minimal temporal association. Single-frame candidates stay visible.

A confirmed track is a repeated observation. It is not a danger claim and
not proof that the corridor is the train path. A missed frame is unknown,
not a false object. Dropping an old track stops tracking; it does not clear
the corridor.
"""

from __future__ import annotations

import numpy as np


PRIOR_FRAME = "sensor_working_prior_mount_unconfirmed"


def _temporal(config: dict) -> dict:
    block = config.get("temporal") or {}
    return {
        "enabled": bool(block.get("enabled", True)),
        "confirm_frames": int(block.get("confirm_frames", 2)),
        "confirm_window": int(block.get("confirm_window", 3)),
        "max_missed_frames": int(block.get("max_missed_frames", 1)),
        "max_track_age_s": float(block.get("max_track_age_s", 0.5)),
        "assumption": str(block.get("assumption") or ""),
    }


def _distance(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _into_current(position, pose_then, pose_now, coordinate_frame):
    """Move s,l,h only in the unconfirmed sensor prior. Not a calibrated frame."""
    if coordinate_frame != PRIOR_FRAME or pose_then is None or pose_now is None:
        return None
    transform = np.linalg.inv(np.asarray(pose_now, dtype=np.float64)) @ np.asarray(pose_then, dtype=np.float64)
    sensor = np.array([float(position["l"]), -float(position["s"]), float(position["h"]), 1.0])
    moved = transform @ sensor
    return [-float(moved[1]), float(moved[0]), float(moved[2])]


class TrackSession:
    def __init__(self, config: dict):
        self.settings = _temporal(config)
        self.link_m = float(config["link_distance_m"]["value"])
        self.tracks = []
        self._next_id = 1
        self.frame_keys = []
        self.applied = set()
        self.coordinate_frame = None

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1
        self.frame_keys = []
        self.applied = set()
        self.coordinate_frame = None

    def update(self, candidates, frame_key, pose=None, registration_reliable=None, coordinate_frame=None, stamp_ns=None):
        """Attach observations. Does not hide a candidate seen on this frame.

        A repeated hypothesis name is not a stable corridor. Linking needs an
        explicit corridor_key. The s=-Y, l=X, h=Z map is only the sensor prior.
        """
        if frame_key in self.applied:
            published = [self._fresh(candidate, "repeat_message_window_unchanged") for candidate in candidates]
            return published, self._state("repeat_message")
        self.applied.add(frame_key)
        self.frame_keys.append(frame_key)
        if self.coordinate_frame is not None and coordinate_frame != self.coordinate_frame:
            self.tracks = []
        self.coordinate_frame = coordinate_frame
        published = []
        if not self.settings["enabled"]:
            for candidate in candidates:
                published.append(self._fresh(candidate, "temporal_disabled"))
            return published, self._state("disabled")
        if pose is None or coordinate_frame != PRIOR_FRAME:
            self._age(stamp_ns)
            reason = "no_pose_sensor_coordinates_not_linked" if pose is None else "coordinate_frame_is_not_the_sensor_prior"
            for candidate in candidates:
                published.append(self._fresh(candidate, reason))
            return published, self._state("association_unavailable")
        if registration_reliable is not True:
            self._age(stamp_ns)
            for candidate in candidates:
                published.append(self._fresh(candidate, "registration_not_reliable"))
            return published, self._state("registration_not_reliable")
        located = []
        for track in self.tracks:
            current = _into_current(track["position_m"], track["pose"], pose, coordinate_frame)
            if current is None:
                track["status"] = "unknown"
                continue
            located.append((track, current))
        pairs = []
        for cand_i, candidate in enumerate(candidates):
            key = candidate.get("corridor_key")
            if not key:
                continue
            position = [candidate["position_m"]["s"], candidate["position_m"]["l"], candidate["position_m"]["h"]]
            for track, current in located:
                if track.get("corridor_key") != key:
                    continue
                distance = _distance(position, current)
                if distance <= self.link_m:
                    pairs.append((distance, cand_i, track["track_id"]))
        by_candidate = {}
        by_track = {}
        for _distance_m, cand_i, track_id in pairs:
            by_candidate.setdefault(cand_i, set()).add(track_id)
            by_track.setdefault(track_id, set()).add(cand_i)
        ambiguous_candidates = {cand_i for cand_i, ids in by_candidate.items() if len(ids) != 1}
        ambiguous_tracks = {track_id for track_id, ids in by_track.items() if len(ids) != 1}
        used = set()
        opened = set()
        for candidate_index, candidate in enumerate(candidates):
            ids = by_candidate.get(candidate_index, set())
            if not candidate.get("corridor_key"):
                published.append(self._fresh(candidate, "hypothesis_name_is_not_a_stable_corridor"))
                continue
            if candidate_index in ambiguous_candidates or any(track_id in ambiguous_tracks for track_id in ids):
                published.append(self._fresh(candidate, "ambiguous_association"))
                continue
            if len(ids) == 1:
                track = next(item for item in self.tracks if item["track_id"] == next(iter(ids)))
                self._confirm(track, candidate, frame_key, pose, stamp_ns)
                used.add(track["track_id"])
                published.append(self._view(track, "linked"))
                continue
            published.append(self._open(candidate, frame_key, pose, stamp_ns))
            opened.add(self.tracks[-1]["track_id"])
        for track in self.tracks:
            if track["track_id"] in used or track["track_id"] in opened or track["status"] == "expired":
                continue
            track["missed"] = int(track["missed"]) + 1
            track["seen"].append(False)
            track["status"] = "expired" if track["missed"] > self.settings["max_missed_frames"] else "missed"
        self.tracks = [track for track in self.tracks if track["status"] != "expired"]
        return published, self._state("active")

    def _fresh(self, candidate, association):
        return {
            "candidate_id": candidate["candidate_id"],
            "hypothesis_id": candidate["hypothesis_id"],
            "track_id": None,
            "confirmation_count": 1,
            "confirmed": False,
            "association": association,
            "single_frame_published": True,
            "confirmed_is_not_danger": True,
            "confirmed_is_not_own_path": True,
        }

    def _age(self, stamp_ns):
        kept = []
        for track in self.tracks:
            track["missed"] = int(track["missed"]) + 1
            old = track.get("stamp_ns")
            too_old = (
                old is not None
                and stamp_ns is not None
                and (int(stamp_ns) - int(old)) / 1e9 > self.settings["max_track_age_s"]
            )
            if track["missed"] > self.settings["max_missed_frames"] or too_old:
                track["status"] = "expired"
            else:
                track["status"] = "missed"
                kept.append(track)
        self.tracks = kept

    def _open(self, candidate, frame_key, pose, stamp_ns):
        track = {
            "track_id": f"T{self._next_id}",
            "candidate_id": candidate["candidate_id"],
            "hypothesis_id": candidate["hypothesis_id"],
            "corridor_key": candidate.get("corridor_key"),
            "stamp_ns": None if stamp_ns is None else int(stamp_ns),
            "position_m": dict(candidate["position_m"]),
            "pose": None if pose is None else np.array(pose, copy=True),
            "frame_keys": [frame_key],
            "seen": [True],
            "missed": 0,
            "status": "active",
        }
        self._next_id += 1
        self.tracks.append(track)
        return self._view(track, "new")

    def _confirm(self, track, candidate, frame_key, pose, stamp_ns):
        track["candidate_id"] = candidate["candidate_id"]
        if stamp_ns is not None:
            track["stamp_ns"] = int(stamp_ns)
        if frame_key not in track["frame_keys"]:
            track["frame_keys"].append(frame_key)
            track["seen"].append(True)
        track["position_m"] = dict(candidate["position_m"])
        track["pose"] = None if pose is None else np.array(pose, copy=True)
        track["missed"] = 0
        track["status"] = "active"
        window = self.settings["confirm_window"]
        track["seen"] = track["seen"][-window:]
        track["frame_keys"] = track["frame_keys"][-window:]

    def _view(self, track, association):
        count = int(sum(bool(item) for item in track["seen"]))
        return {
            "candidate_id": track.get("candidate_id"),
            "hypothesis_id": track["hypothesis_id"],
            "track_id": track["track_id"],
            "confirmation_count": count,
            "confirmed": count >= self.settings["confirm_frames"],
            "association": association,
            "status": track["status"],
            "single_frame_published": True,
            "missed_is_not_a_false_object": True,
            "expired_is_not_free_space": True,
            "confirmed_is_not_danger": True,
            "confirmed_is_not_own_path": True,
            "confirmation_is_an_engineering_assumption": True,
            "assumption": self.settings["assumption"],
        }

    def _state(self, mode):
        return {
            "mode": mode,
            "tracks": [
                {
                    "track_id": track["track_id"],
                    "hypothesis_id": track["hypothesis_id"],
                    "status": track["status"],
                    "confirmation_count": int(sum(bool(item) for item in track["seen"])),
                    "expired_is_not_free_space": True,
                }
                for track in self.tracks
            ],
            "kiss_coarse_gate_is_not_centimetre_accuracy": True,
        }
