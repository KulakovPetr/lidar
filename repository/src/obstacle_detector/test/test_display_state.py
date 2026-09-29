import unittest

from obstacle_detector.display_state import annotate_display, previous_marker_text, processed_marker_text


class DisplayStateTests(unittest.TestCase):
    def test_delayed_frame_is_not_called_the_live_scene(self):
        record = annotate_display(
            {"status": "ok", "decision": "conditional_intrusion"},
            processing_id=4, session_id=2, stamp_ns=99, frame_id="hesai_lidar",
            superseded=True, received_now=9, reception_index=4,
        )
        self.assertFalse(record["result_is_current_scene"])
        self.assertEqual(record["result_age_receptions"], 5)
        text = processed_marker_text(record)
        self.assertIn("условное вторжение в заданный профиль", text)
        self.assertIn("processing 4", text)
        self.assertIn("session 2", text)
        self.assertIn("не текущее состояние пространства", text)

    def test_timeout_is_an_error_and_previous_is_labeled(self):
        record = annotate_display(
            {"status": "processing_timeout", "decision": "processing_timeout"},
            processing_id=5, session_id=1, stamp_ns=10, frame_id="hesai_lidar",
            superseded=False, received_now=2, reception_index=1,
        )
        self.assertEqual(record["display_role"], "processing_error")
        self.assertIn("ошибка обработки", processed_marker_text(record))
        previous = {"session_id": 1, "header_stamp_ns": 7, "processing_id": 3}
        self.assertIn("предыдущий результат", previous_marker_text(previous))
        self.assertIn("не ответ для кадра с ошибкой", previous_marker_text(previous))


if __name__ == "__main__":
    unittest.main()