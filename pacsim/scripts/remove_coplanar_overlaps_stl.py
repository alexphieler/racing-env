#!/usr/bin/env python3
"""
Remove high-confidence coplanar triangle overlaps from a binary STL.

This is meant for render z-fighting cleanup, not manifold reconstruction. It
keeps the original triangle coordinates and only deletes faces whose projected
area is almost completely covered by other triangles on the same plane.
"""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def load_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")

    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * tri_count
    if len(data) != expected:
        raise ValueError(
            f"{path} does not look like a binary STL: "
            f"expected {expected} bytes from header, got {len(data)}"
        )

    triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    for i in range(tri_count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles[i] = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
        offset += 50
    return triangles


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    header = b"coplanar-overlap-cleaned".ljust(80, b" ")
    out = bytearray(header)
    out.extend(struct.pack("<I", len(triangles)))

    normals, _areas = triangle_normals_and_areas(triangles)
    for normal, tri in zip(normals, triangles):
        out.extend(
            struct.pack(
                "<12fH",
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
                float(tri[0, 0]),
                float(tri[0, 1]),
                float(tri[0, 2]),
                float(tri[1, 0]),
                float(tri[1, 1]),
                float(tri[1, 2]),
                float(tri[2, 0]),
                float(tri[2, 1]),
                float(tri[2, 2]),
                0,
            )
        )

    path.write_bytes(out)


def triangle_normals_and_areas(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = lengths > 1e-12
    normals[valid] = cross[valid] / lengths[valid, None]
    return normals, lengths * 0.5


def canonical_plane_keys(
    triangles: np.ndarray,
    normals: np.ndarray,
    normal_quant: float,
    plane_quant: float,
) -> np.ndarray:
    canonical_normals = normals.copy()
    offsets = np.einsum("ij,ij->i", canonical_normals, triangles[:, 0])

    for i, normal in enumerate(canonical_normals):
        axis = int(np.argmax(np.abs(normal)))
        if normal[axis] < 0.0:
            canonical_normals[i] *= -1.0
            offsets[i] *= -1.0

    return np.concatenate(
        [
            np.round(canonical_normals / normal_quant).astype(np.int32),
            np.round(offsets[:, None] / plane_quant).astype(np.int32),
        ],
        axis=1,
    )


def projected_triangle_polygon(triangle: np.ndarray, axes: list[int]) -> Polygon:
    polygon = Polygon(triangle[:, axes])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def find_overlaps(
    triangles: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    normal_quant: float,
    plane_quant: float,
    pair_overlap: float,
    coverage_overlap: float,
) -> tuple[list[tuple[int, int, float, float]], np.ndarray, list[list[object]]]:
    keys = canonical_plane_keys(triangles, normals, normal_quant, plane_quant)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for face_index, key in enumerate(map(tuple, keys)):
        if areas[face_index] > 1e-12:
            groups[key].append(face_index)

    coverage_pieces: list[list[object]] = [[] for _ in range(len(triangles))]
    pairs: list[tuple[int, int, float, float]] = []

    for indices in groups.values():
        if len(indices) < 2:
            continue

        group_normal = np.mean(normals[indices], axis=0)
        drop_axis = int(np.argmax(np.abs(group_normal)))
        axes = [axis for axis in range(3) if axis != drop_axis]

        polygons = [projected_triangle_polygon(triangles[i], axes) for i in indices]
        bounds = [
            polygon.bounds if not polygon.is_empty else (0.0, 0.0, 0.0, 0.0)
            for polygon in polygons
        ]

        for a in range(len(indices)):
            polygon_a = polygons[a]
            if polygon_a.is_empty:
                continue
            bounds_a = bounds[a]

            for b in range(a + 1, len(indices)):
                polygon_b = polygons[b]
                if polygon_b.is_empty:
                    continue
                bounds_b = bounds[b]
                if (
                    bounds_a[2] < bounds_b[0]
                    or bounds_b[2] < bounds_a[0]
                    or bounds_a[3] < bounds_b[1]
                    or bounds_b[3] < bounds_a[1]
                ):
                    continue

                intersection = polygon_a.intersection(polygon_b)
                if intersection.is_empty:
                    continue

                intersection_area = intersection.area
                if intersection_area <= 1e-10:
                    continue

                smaller_area = max(min(polygon_a.area, polygon_b.area), 1e-12)
                overlap_ratio = intersection_area / smaller_area
                face_a = indices[a]
                face_b = indices[b]

                if overlap_ratio >= coverage_overlap:
                    coverage_pieces[face_a].append(intersection)
                    coverage_pieces[face_b].append(intersection)

                if overlap_ratio >= pair_overlap:
                    pairs.append((face_a, face_b, overlap_ratio, intersection_area))

    coverage = np.zeros(len(triangles), dtype=np.float64)
    for face_index, pieces in enumerate(coverage_pieces):
        if not pieces:
            continue
        try:
            covered = unary_union(pieces).area
        except Exception:
            covered = sum(piece.area for piece in pieces)
        coverage[face_index] = min(covered / max(areas[face_index], 1e-12), 1.0)

    return pairs, coverage, coverage_pieces


def choose_removed_faces(
    triangles: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    pairs: list[tuple[int, int, float, float]],
    coverage: np.ndarray,
    covered_threshold: float,
    tie_break: str,
) -> set[int]:
    center = (triangles.reshape(-1, 3).min(axis=0) + triangles.reshape(-1, 3).max(axis=0)) * 0.5
    centroids = triangles.mean(axis=1)
    outward = np.einsum("ij,ij->i", normals, centroids - center)

    removed: set[int] = set()
    pairs_by_strength = sorted(pairs, key=lambda item: (item[2], item[3]), reverse=True)

    for face_a, face_b, _ratio, _area in pairs_by_strength:
        if face_a in removed or face_b in removed:
            continue

        candidate_a = coverage[face_a] >= covered_threshold
        candidate_b = coverage[face_b] >= covered_threshold
        if not candidate_a and not candidate_b:
            continue

        if candidate_a and not candidate_b:
            removed.add(face_a)
            continue
        if candidate_b and not candidate_a:
            removed.add(face_b)
            continue

        outward_delta = outward[face_a] - outward[face_b]
        if abs(outward_delta) > 1e-5:
            removed.add(face_a if outward_delta < 0.0 else face_b)
            continue

        if tie_break == "keep-smaller":
            removed.add(face_a if areas[face_a] > areas[face_b] else face_b)
        else:
            removed.add(face_a if areas[face_a] <= areas[face_b] else face_b)

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--normal-quant",
        type=float,
        default=0.001,
        help="Normal quantization for coplanar buckets. Default: 0.001",
    )
    parser.add_argument(
        "--plane-quant",
        type=float,
        default=0.02,
        help="Plane offset quantization in STL units. With xacro scale 0.01, 0.02 is 0.2 mm. Default: 0.02",
    )
    parser.add_argument(
        "--pair-overlap",
        type=float,
        default=0.95,
        help="Only resolve face pairs where the intersection covers this fraction of the smaller face. Default: 0.95",
    )
    parser.add_argument(
        "--coverage-overlap",
        type=float,
        default=0.05,
        help="Pair overlap threshold used while computing per-face covered area. Default: 0.05",
    )
    parser.add_argument(
        "--covered-threshold",
        type=float,
        default=0.99,
        help="A face is removable only when this fraction of its area is covered. Default: 0.99",
    )
    parser.add_argument(
        "--tie-break",
        choices=("keep-smaller", "keep-larger"),
        default="keep-smaller",
        help="Choice when two redundant faces are equally exterior. Default: keep-smaller",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing output")
    args = parser.parse_args()

    triangles = load_binary_stl(args.input)
    normals, areas = triangle_normals_and_areas(triangles)

    degenerate = areas <= 1e-12
    pairs, coverage, _coverage_pieces = find_overlaps(
        triangles,
        normals,
        areas,
        args.normal_quant,
        args.plane_quant,
        args.pair_overlap,
        args.coverage_overlap,
    )
    removed = choose_removed_faces(
        triangles,
        normals,
        areas,
        pairs,
        coverage,
        args.covered_threshold,
        args.tie_break,
    )
    removed.update(np.flatnonzero(degenerate).tolist())

    remaining_pair_count = sum(1 for a, b, _ratio, _area in pairs if a not in removed and b not in removed)
    high_coverage_count = int(np.count_nonzero(coverage >= args.covered_threshold))

    print(f"input triangles: {len(triangles)}")
    print(f"pair overlaps >= {args.pair_overlap:.3f}: {len(pairs)}")
    print(f"faces covered >= {args.covered_threshold:.3f}: {high_coverage_count}")
    print(f"removed triangles: {len(removed)}")
    print(f"remaining unresolved high-overlap pairs: {remaining_pair_count}")

    if args.dry_run:
        return

    keep_mask = np.ones(len(triangles), dtype=bool)
    keep_mask[list(removed)] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(args.output, triangles[keep_mask])
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
