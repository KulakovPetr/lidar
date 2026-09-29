"""Labels for the last processed frame. This does not change detection."""

from __future__ import annotations


def annotate_display(
    record: dict,
    *,
    processing_id: int,
    session_id: int,
    stamp_ns: int,
    frame_id: str,
    superseded: bool,
    received_now: int,
    reception_index: int,
) -> dict:
    """Mark a finished frame as a delayed result, not the live scene."""
    record["processing_id"] = int(processing_id)
    record["session_id"] = int(session_id)
    record["header_stamp_ns"] = int(stamp_ns)
    record["header_frame_id"] = frame_id
    record["belongs_to_header_stamp_ns"] = int(stamp_ns)
    record["result_is_current_scene"] = False
    record["result_is_last_processed"] = True
    record["newer_input_exists"] = bool(superseded)
    record["result_age_receptions"] = int(received_now) - int(reception_index)
    record["display_mode"] = "ros_processing_with_delay"
    if record.get("status") == "processing_timeout":
        record["display_role"] = "processing_error"
        record["display_label"] = "таймаут этого кадра"
        record["timeout_is_not_absence_of_obstacle"] = True
        record["show_previous_result"] = False
    elif record.get("status") == "processing_error":
        record["display_role"] = "processing_error"
        record["display_label"] = "ошибка обработки"
    else:
        record["display_role"] = "last_processed"
        record["display_label"] = "последний обработанный кадр"
    return record


def processed_marker_text(record: dict) -> str:
    """One card. The same processing_id is on the cloud and the markers."""
    decision = record.get("decision") or record.get("status") or ""
    if decision == "conditional_intrusion":
        title = "условное вторжение в заданный профиль"
    elif decision == "candidate_only":
        title = "кандидат без условного вторжения"
    elif record.get("display_role") == "processing_error":
        title = "ошибка обработки"
    else:
        title = str(decision)
    return (
        f"{title}\n"
        f"{record.get('display_label')}\n"
        f"session {record.get('session_id')} stamp {record.get('header_stamp_ns')} "
        f"processing {record.get('processing_id')}\n"
        f"задержка, приёмов: {record.get('result_age_receptions')}\n"
        "не текущее состояние пространства"
    )


def previous_marker_text(record: dict) -> str:
    return (
        "предыдущий результат\n"
        f"session {record.get('session_id')} stamp {record.get('header_stamp_ns')} "
        f"processing {record.get('processing_id')}\n"
        "не ответ для кадра с ошибкой обработки"
    )
