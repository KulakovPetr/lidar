"""Read one PointCloud2 field without assuming an XYZ dtype.

Row padding, endianness, datatype, and offset are taken from the message.
A float64 at a non-aligned offset is gathered byte by byte.
"""

from __future__ import annotations

import numpy as np

INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = 1, 2, 3, 4, 5, 6, 7, 8
DTYPE_BY_CODE = {
    INT8: np.int8,
    UINT8: np.uint8,
    INT16: np.int16,
    UINT16: np.uint16,
    INT32: np.int32,
    UINT32: np.uint32,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
}
DTYPE_NAME = {
    INT8: "int8",
    UINT8: "uint8",
    INT16: "int16",
    UINT16: "uint16",
    INT32: "int32",
    UINT32: "uint32",
    FLOAT32: "float32",
    FLOAT64: "float64",
}
RETURN_FIELD_NAMES = frozenset(
    {"return", "return_type", "return_mode", "echo", "dual_return"}
)


class FieldSpec:
    def __init__(self, name: str, offset: int, datatype: int, count: int) -> None:
        self.name = name
        self.offset = int(offset)
        self.datatype = int(datatype)
        self.count = int(count)


def extract_field(raw, height, width, point_step, row_step, offset, datatype, is_bigendian):
    """Return a 1-d array of one scalar field."""
    if datatype not in DTYPE_BY_CODE:
        raise ValueError(f"unknown PointField datatype {datatype}")
    host = np.dtype(DTYPE_BY_CODE[datatype])
    dt = host.newbyteorder(">" if is_bigendian else "<")
    item = int(dt.itemsize)
    if offset < 0 or int(point_step) < offset + item:
        raise ValueError(
            f"field offset {offset} size {item} does not fit point_step {point_step}"
        )
    n = int(height) * int(width)
    tight = int(row_step) == int(point_step) * int(width)
    align = max(int(dt.alignment), 1)
    if tight and offset % align == 0 and int(point_step) % align == 0:
        return np.ndarray(
            shape=(n,), dtype=dt, buffer=raw, offset=int(offset), strides=(int(point_step),)
        )
    rows = np.repeat(np.arange(int(height), dtype=np.int64), int(width))
    cols = np.tile(np.arange(int(width), dtype=np.int64), int(height))
    starts = rows * int(row_step) + cols * int(point_step) + int(offset)
    gathered = np.empty((n, item), dtype=np.uint8)
    for byte_i in range(item):
        gathered[:, byte_i] = raw[starts + byte_i]
    return gathered.view(dt).reshape(n)


def decode_cloud(raw, height, width, point_step, row_step, is_bigendian, fields):
    """Decode every scalar field. XYZ dtype is whatever the field declares."""
    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
        buffer = raw
    else:
        buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
    decoded = {}
    layout = []
    for field in fields:
        if field.count != 1:
            raise ValueError(f"field {field.name} count {field.count} is not a scalar")
        values = extract_field(
            buffer,
            height,
            width,
            point_step,
            row_step,
            field.offset,
            field.datatype,
            is_bigendian,
        )
        decoded[field.name] = values
        layout.append(
            {
                "name": field.name,
                "offset": field.offset,
                "datatype": DTYPE_NAME.get(field.datatype, str(field.datatype)),
                "count": field.count,
            }
        )
    for axis in ("x", "y", "z"):
        if axis not in decoded:
            raise ValueError(f"cloud has no {axis} field")
        if not np.issubdtype(decoded[axis].dtype, np.floating) or decoded[axis].dtype.itemsize not in (4, 8):
            raise ValueError(f"{axis} datatype is {decoded[axis].dtype}, expected float32 or float64")
    xyz_dtypes = {decoded[axis].dtype for axis in ("x", "y", "z")}
    if len(xyz_dtypes) != 1:
        raise ValueError(f"XYZ dtypes differ: {xyz_dtypes}")
    return decoded, layout


def return_type_field(layout) -> str | None:
    for field in layout:
        if field["name"].lower() in RETURN_FIELD_NAMES:
            return field["name"]
    return None
