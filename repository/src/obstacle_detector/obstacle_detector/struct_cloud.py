"""Read PointCloud2 fields with struct.

This module does not call the numpy gather used by decode_cloud.
"""

from __future__ import annotations

import struct

_STRUCT_CODE = {
    1: "b",
    2: "B",
    3: "h",
    4: "H",
    5: "i",
    6: "I",
    7: "f",
    8: "d",
}


def read_points_struct(raw, height, width, point_step, row_step, is_bigendian, fields, indices):
    """Return one dict per index. `fields` are (name, offset, datatype, count)."""
    if isinstance(raw, bytes):
        blob = raw
    elif isinstance(raw, bytearray):
        blob = bytes(raw)
    elif hasattr(raw, "tobytes"):
        blob = raw.tobytes()
    else:
        blob = bytes(raw)
    endian = ">" if is_bigendian else "<"
    readers = []
    for name, offset, datatype, count in fields:
        if int(count) != 1:
            raise ValueError(f"{name} count {count} is not scalar")
        code = _STRUCT_CODE.get(int(datatype))
        if code is None:
            raise ValueError(f"unknown datatype {datatype}")
        fmt = endian + code
        readers.append((name, int(offset), fmt, struct.calcsize(fmt)))
    height = int(height)
    width = int(width)
    point_step = int(point_step)
    row_step = int(row_step)
    n = height * width
    rows = []
    for index in indices:
        index = int(index)
        if index < 0 or index >= n:
            raise IndexError(index)
        row = index // width
        col = index - row * width
        base = row * row_step + col * point_step
        record = {"index": index}
        for name, offset, fmt, size in readers:
            start = base + offset
            record[name] = struct.unpack_from(fmt, blob, start)[0]
            if start + size > len(blob):
                raise ValueError(f"field {name} at {start} exceeds buffer")
        rows.append(record)
    return rows
