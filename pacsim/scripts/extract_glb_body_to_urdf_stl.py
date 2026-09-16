"""Extract the GLB car body mesh into the URDF STL coordinate convention.

The pacsim URDF body mesh is authored in raw STL units and then scaled by
0.01 in separate_model.xacro. Blender/GLB exports often bake a different axis
or scale convention, which can make the body invisible or oversized when used
as a direct STL replacement.

This script reads the body mesh named "Model_without_Steering_Tires" from
pipeline/Models/car.glb, converts GLB coordinates (x, y, z) to the URDF STL
body convention (x, z, y), and writes a binary STL that should be used with:

    <mesh filename="package://pacsim/urdf/Model_without_Steering_Tires.stl"
          scale="0.01 0.01 0.01"/>

It has no third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


COMPONENT_DTYPE = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/root/workspace/pipeline/Models/car.glb",
        help="Source GLB containing Model_without_Steering_Tires.",
    )
    parser.add_argument(
        "--output",
        default="/root/workspace/pacsim/urdf/Model_without_Steering_Tires_from_glb.stl",
        help="Destination binary STL.",
    )
    parser.add_argument(
        "--mesh-name",
        default="Model_without_Steering_Tires",
        help="GLB mesh name to extract.",
    )
    return parser.parse_args()


def load_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        magic, version, length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2:
            raise RuntimeError(f"{path} is not a GLB v2 file")

        json_chunk = None
        bin_chunk = None
        while handle.tell() < length:
            chunk_length, chunk_type = struct.unpack("<I4s", handle.read(8))
            chunk = handle.read(chunk_length)
            if chunk_type == b"JSON":
                json_chunk = json.loads(chunk.decode("utf-8").rstrip("\x00 \n\r\t"))
            elif chunk_type == b"BIN\0":
                bin_chunk = chunk

    if json_chunk is None or bin_chunk is None:
        raise RuntimeError(f"{path} is missing JSON or BIN chunks")
    return json_chunk, bin_chunk


def read_accessor(gltf: dict[str, Any], bin_chunk: bytes, accessor_index: int) -> list[tuple[float, ...] | int]:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    fmt_char, component_size = COMPONENT_DTYPE[accessor["componentType"]]
    component_count = TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    stride = view.get("byteStride", component_count * component_size)
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    fmt = "<" + (fmt_char * component_count)

    values: list[tuple[float, ...] | int] = []
    for row in range(count):
        item = struct.unpack_from(fmt, bin_chunk, offset + row * stride)
        if component_count == 1:
            values.append(item[0])
        else:
            values.append(tuple(float(v) for v in item))
    return values


def convert_position_glb_to_urdf_stl(p: tuple[float, ...]) -> tuple[float, float, float]:
    return (float(p[0]), float(p[2]), float(p[1]))


def normal_for_triangle(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-20:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    gltf, bin_chunk = load_glb(source)
    mesh_index = None
    for index, mesh in enumerate(gltf.get("meshes", [])):
        if mesh.get("name") == args.mesh_name:
            mesh_index = index
            break
    if mesh_index is None:
        raise RuntimeError(f"mesh '{args.mesh_name}' not found in {source}")

    mesh = gltf["meshes"][mesh_index]
    if len(mesh.get("primitives", [])) != 1:
        raise RuntimeError(f"mesh '{args.mesh_name}' must contain exactly one primitive")
    primitive = mesh["primitives"][0]
    if primitive.get("mode", 4) != 4:
        raise RuntimeError(f"mesh '{args.mesh_name}' is not a triangle-list primitive")

    positions_raw = read_accessor(gltf, bin_chunk, primitive["attributes"]["POSITION"])
    indices_raw = read_accessor(gltf, bin_chunk, primitive["indices"])
    positions = [convert_position_glb_to_urdf_stl(p) for p in positions_raw]  # type: ignore[arg-type]
    indices = [int(i) for i in indices_raw]
    if len(indices) % 3 != 0:
        raise RuntimeError("index count is not divisible by 3")

    triangle_count = len(indices) // 3
    header = b"Extracted from car.glb Model_without_Steering_Tires for pacsim URDF"
    header = header[:80].ljust(80, b"\0")
    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", triangle_count))
        for tri_start in range(0, len(indices), 3):
            a = positions[indices[tri_start + 0]]
            b = positions[indices[tri_start + 1]]
            c = positions[indices[tri_start + 2]]
            n = normal_for_triangle(a, b, c)
            handle.write(struct.pack("<12fH", *(n + a + b + c), 0))

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    print(f"wrote {destination}")
    print(f"triangles={triangle_count}")
    print(f"bbox_min=({min(xs):.6g}, {min(ys):.6g}, {min(zs):.6g})")
    print(f"bbox_max=({max(xs):.6g}, {max(ys):.6g}, {max(zs):.6g})")


if __name__ == "__main__":
    main()
