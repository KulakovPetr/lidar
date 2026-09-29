from obstacle_detector.ros_node import display_slot_accepts, picture_update


def test_late_frame_does_not_replace_the_picture():
    current = {"session": 1, "reception": 5, "status": "ok"}
    assert display_slot_accepts(current, 1, 4) is False
    slot, note = picture_update(current, "ok", 1, 4)
    assert slot is current
    assert note is None


def test_newer_frame_replaces_the_picture():
    current = {"session": 1, "reception": 5, "status": "ok"}
    slot, note = picture_update(current, "ok", 1, 6)
    assert slot is not current
    assert slot["reception"] == 6
    assert note is None


def test_timeout_keeps_the_cloud_and_notes_the_result():
    current = {"session": 1, "reception": 5, "status": "ok"}
    slot, note = picture_update(current, "processing_timeout", 1, 6)
    assert slot is current
    assert note["status"] == "processing_timeout"
    assert note["reception"] == 6


def test_new_session_replaces_an_older_session():
    current = {"session": 1, "reception": 9, "status": "ok"}
    assert display_slot_accepts(current, 2, 1) is True
    assert display_slot_accepts(current, 1, 1) is False
