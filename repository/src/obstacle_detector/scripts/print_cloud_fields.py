"""Print PointCloud2 header fields for one development bag.

Does not detect obstacles and does not open the frozen run.
"""

from __future__ import annotations

import sys

HELD_OUT_MARKER = "doubleT_obstacle"


def refuse_held_out(bag_uri: str) -> None:
    if HELD_OUT_MARKER in bag_uri.replace("\\", "/"):
        raise SystemExit(
            f"refusing to read frozen run {HELD_OUT_MARKER}: {bag_uri}"
        )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: print_cloud_fields.py BAG_DIR", file=sys.stderr)
        return 2
    bag_uri = argv[1]
    refuse_held_out(bag_uri)

    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import PointCloud2

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topics = reader.get_all_topics_and_types()
    print("topics:")
    for topic in topics:
        print(f"  {topic.name}  {topic.type}")

    if not reader.has_next():
        print("no messages")
        return 1
    topic_name, payload, stamp_ns = reader.read_next()
    cloud = deserialize_message(payload, PointCloud2)

    fields = ",".join(f"{field.name}:{field.datatype}@{field.offset}" for field in cloud.fields)
    print(f"topic={topic_name}")
    print(f"stamp_ns={stamp_ns}")
    print(f"frame_id={cloud.header.frame_id}")
    print(f"height={cloud.height}")
    print(f"width={cloud.width}")
    print(f"point_step={cloud.point_step}")
    print(f"row_step={cloud.row_step}")
    print(f"is_dense={cloud.is_dense}")
    print(f"is_bigendian={cloud.is_bigendian}")
    print(f"fields={fields}")
    print(f"data_bytes={len(cloud.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
