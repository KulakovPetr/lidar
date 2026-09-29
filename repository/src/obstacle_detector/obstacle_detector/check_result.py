"""Compare two offline results. Time and machine notes are not algorithm output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


RANGE_TOLERANCE_M = 1e-5


def _load(path: Path) -> tuple[dict[int, dict], list[str]]:
    found: dict[int, dict] = {}
    errors: list[str] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}, [f"пустой результат {path}"]
    for line in lines:
        record = json.loads(line)
        index = int(record["frame_index"])
        if index in found:
            errors.append(f"дубль индекса {index}")
            continue
        found[index] = record
    return found, errors


def _membership(record: dict) -> list:
    rows = []
    for item in record.get("candidates") or []:
        indices = tuple(sorted(int(value) for value in item.get("source_indices") or []))
        rows.append((item.get("hypothesis_id"), item.get("candidate_id"), item.get("kind"), item.get("branch"), indices, item.get("range_from_lidar_m")))
    return rows


def _hypotheses(record: dict) -> list:
    rows = []
    for item in (record.get("geometry_quality") or {}).get("hypotheses") or []:
        rows.append((item.get("hypothesis_id"), item.get("kind"), item.get("geometry_applicable"), item.get("observed_s_min_m"), item.get("observed_s_max_m")))
    return rows


def _finite_range(index: int, value) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{index}: дальность не является числом"
    if not math.isfinite(number):
        return f"{index}: дальность не конечна"
    return None


def compare(reference: Path, actual: Path, tolerance_m: float = RANGE_TOLERANCE_M) -> list[str]:
    left, left_errors = _load(reference)
    right, right_errors = _load(actual)
    errors = list(left_errors) + list(right_errors)
    if not left or not right:
        return errors
    if set(left) != set(right):
        errors.append(f"неожиданный набор сообщений {sorted(set(left) ^ set(right))}")
    for index in sorted(set(left) & set(right)):
        a = left[index]
        b = right[index]
        if a.get("status") != b.get("status"):
            errors.append(f"{index}: status {a.get('status')} != {b.get('status')}")
        if a.get("coordinate_frame") != b.get("coordinate_frame"):
            errors.append(f"{index}: coordinate_frame {a.get('coordinate_frame')} != {b.get('coordinate_frame')}")
        if a.get("header_stamp_ns") != b.get("header_stamp_ns") or a.get("header_frame_id") != b.get("header_frame_id"):
            errors.append(f"{index}: исходные timestamp или frame_id различаются")
        if a.get("config_version") != b.get("config_version"):
            errors.append(f"{index}: эффективная конфигурация различается")
        for record in (a, b):
            for item in record.get("candidates") or []:
                problem = _finite_range(index, item.get("range_from_lidar_m"))
                if problem:
                    errors.append(problem)
                    break
        if _hypotheses(a) != _hypotheses(b):
            errors.append(f"{index}: гипотезы различаются")
        am = _membership(a)
        bm = _membership(b)
        if len(am) != len(bm):
            errors.append(f"{index}: кандидатов {len(am)} != {len(bm)}")
            continue
        for one, two in zip(am, bm):
            if one[:5] != two[:5]:
                errors.append(f"{index}: состав кандидата {one[1]} различается")
                break
            if one[5] is None or two[5] is None:
                if one[5] != two[5]:
                    errors.append(f"{index}: дальность {one[5]} != {two[5]}")
                    break
                continue
            if abs(float(one[5]) - float(two[5])) > tolerance_m:
                errors.append(f"{index}: дальность {one[5]} и {two[5]} дальше допуска {tolerance_m} м")
                break
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнить два результата offline-прогона")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--tolerance-m", type=float, default=RANGE_TOLERANCE_M)
    args = parser.parse_args(argv)
    errors = compare(args.reference, args.actual, args.tolerance_m)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(json.dumps({"match": True, "tolerance_m": args.tolerance_m, "time_and_machine_not_compared": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
