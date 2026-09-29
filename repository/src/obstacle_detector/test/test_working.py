"""Stage 11B checks. Synthetic tracking, not a lidar visibility model."""

import inspect
import unittest

import numpy as np

from obstacle_detector.corridor import classify_sensor_points, load_config, working_to_sensor
from obstacle_detector.detector import detect, local_protrusions, load_detector_config, subsample_grid
from obstacle_detector.tracking import PRIOR_FRAME, TrackSession
from obstacle_detector.working import options_from_config, process_working_frame
from test.test_corridor import _mount
from test.test_detector import _corridor, _detector_config


def _candidate(candidate_id, hypothesis_id, s, l, h, corridor_key=None):
    item = {
        "candidate_id": candidate_id,
        "hypothesis_id": hypothesis_id,
        "position_m": {"s": s, "l": l, "h": h},
    }
    if corridor_key is not None:
        item["corridor_key"] = corridor_key
    return item


def _update(session, candidates, frame_key, pose=None, reliable=True, frame=PRIOR_FRAME, stamp_ns=None):
    return session.update(
        candidates,
        frame_key,
        pose=pose if pose is not None else _pose(0.0),
        registration_reliable=reliable,
        coordinate_frame=frame,
        stamp_ns=stamp_ns,
    )


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


class WorkingTests(unittest.TestCase):
    def test_new_options_default_off(self):
        options = options_from_config(load_detector_config())
        self.assertFalse(options["kiss_icp"])
        self.assertFalse(options["local_context"])

    def test_context_points_do_not_become_candidates(self):
        config = _detector_config()
        current_s = np.linspace(9.0, 11.0, 16)
        current_l = np.zeros(16)
        current_h = np.zeros(16)
        query = np.ones(16, dtype=bool)
        context = (np.full(30, 10.0), np.full(30, 0.0), np.full(30, 5.0))
        found = local_protrusions(current_s, current_l, current_h, query, config, context_slh=context)
        self.assertEqual(found["mask"].shape[0], 16)
        self.assertTrue(found["context_points_are_not_candidates"])

    def test_baseline_detect_ignores_the_new_argument_when_empty(self):
        corridor = _corridor(status="stated")
        points = subsample_grid((15.0 - 0.15, -0.15, -1.0), (0.3, 0.3, 0.1), (4, 4, 3), 48, 4)
        s, l, h = points[:, 0], points[:, 1], points[:, 2]
        x, y, z = working_to_sensor(s, l, h)
        relation = classify_sensor_points(x, y, z, corridor, _mount(), load_config() | {"front_ahead_of_lidar_m": {"value": None}})["geometric_relation"]
        args = (s, l, h, h - (-1.0), np.ones(s.shape, dtype=bool), np.sqrt(x * x + y * y + z * z), np.arange(s.size), relation, corridor, [], 0.02, _detector_config())
        left = detect(*args)
        right = detect(*args, local_context_slh=(np.zeros(0), np.zeros(0), np.zeros(0)))
        self.assertEqual([item["candidate_id"] for item in left["candidates_large"]], [item["candidate_id"] for item in right["candidates_large"]])
        self.assertNotIn("inserted", inspect.signature(detect).parameters)
        self.assertNotIn("inserted", inspect.signature(process_working_frame).parameters)

    def test_stationary_object_links_when_the_sensor_moves(self):
        session = TrackSession(_detector_config())
        first, _state = _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0, pose=_pose(0.0))
        self.assertEqual(first[0]["confirmation_count"], 1)
        self.assertFalse(first[0]["confirmed"])
        second, _state = _update(session, [_candidate("L1", "H", 10.0, -1.0, 0.0, "C")], 1, pose=_pose(1.0))
        self.assertEqual(second[0]["association"], "linked")
        self.assertEqual(second[0]["confirmation_count"], 2)
        self.assertTrue(second[0]["confirmed"])
        self.assertTrue(second[0]["confirmed_is_not_danger"])

    def test_unchanged_sensor_coordinates_do_not_link_a_moving_sensor(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0, pose=_pose(0.0))
        second, _state = _update(session, [_candidate("L2", "H", 10.0, 0.0, 0.0, "C")], 1, pose=_pose(1.0))
        self.assertEqual(second[0]["association"], "new")

    def test_single_observation_and_a_missed_frame(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0, pose=_pose(0.0), stamp_ns=0)
        _published, state = _update(session, [], 1, pose=_pose(0.0), stamp_ns=100_000_000)
        self.assertEqual(state["tracks"][0]["status"], "missed")
        self.assertNotEqual(state["tracks"][0]["status"], "false")

    def test_two_close_objects_stay_separate(self):
        session = TrackSession(_detector_config())
        published, _state = _update(
            session,
            [_candidate("L1", "H", 10.0, 0.0, 0.0, "A"), _candidate("L2", "H", 10.0, 0.5, 0.0, "B")],
            0,
        )
        self.assertEqual({item["track_id"] for item in published}, {"T1", "T2"})

    def test_ambiguous_association_does_not_add_a_confirmation(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0)
        published, _state = _update(
            session,
            [_candidate("A", "H", 10.0, 0.05, 0.0, "C"), _candidate("B", "H", 10.0, -0.05, 0.0, "C")],
            1,
        )
        self.assertTrue(all(item["association"] == "ambiguous_association" for item in published))
        self.assertTrue(all(item["confirmation_count"] == 1 for item in published))

    def test_bad_registration_and_reset(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0, stamp_ns=0)
        published, state = _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 1, reliable=False, stamp_ns=100_000_000)
        self.assertEqual(published[0]["association"], "registration_not_reliable")
        self.assertEqual(published[0]["confirmation_count"], 1)
        self.assertEqual(state["mode"], "registration_not_reliable")
        session.reset()
        self.assertEqual(session.tracks, [])

    def test_same_frame_points_are_one_confirmation(self):
        session = TrackSession(_detector_config())
        published, _state = _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 4)
        again, state = _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 4)
        self.assertEqual(state["mode"], "repeat_message")
        self.assertEqual(len(session.tracks[0]["seen"]), 1)
        self.assertEqual(published[0]["confirmation_count"], 1)
        self.assertEqual(again[0]["confirmation_count"], 1)

    def test_incompatible_hypothesis_is_not_linked(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H1", 10.0, 0.0, 0.0, "left")], 0)
        published, _state = _update(session, [_candidate("L2", "H1", 10.0, 0.0, 0.0)], 1)
        self.assertEqual(published[0]["association"], "hypothesis_name_is_not_a_stable_corridor")
        self.assertFalse(published[0]["confirmed"])

    def test_same_hypothesis_name_does_not_accumulate(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H1", 10.0, 0.0, 0.0)], 0)
        second, _state = _update(session, [_candidate("L1", "H1", 10.0, 0.0, 0.0)], 1)
        self.assertEqual(second[0]["association"], "hypothesis_name_is_not_a_stable_corridor")
        self.assertEqual(second[0]["confirmation_count"], 1)

    def test_pose_loss_expires_old_confirmation(self):
        session = TrackSession(_detector_config())
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 0, stamp_ns=0)
        _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 1, reliable=False, stamp_ns=100_000_000)
        _published, state = _update(session, [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")], 2, reliable=False, stamp_ns=200_000_000)
        self.assertEqual(state["tracks"], [])

    def test_calibrated_frame_does_not_use_the_prior_map(self):
        session = TrackSession(_detector_config())
        published, state = _update(
            session,
            [_candidate("L1", "H", 10.0, 0.0, 0.0, "C")],
            0,
            frame="train_working",
        )
        self.assertEqual(published[0]["association"], "coordinate_frame_is_not_the_sensor_prior")
        self.assertEqual(state["mode"], "association_unavailable")
