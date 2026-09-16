"""Batch-generate candidate repaired car body meshes in Blender.

Run from Blender, for example:

blender --background --python /root/workspace/pacsim/scripts/blender_car_body_repair_batch.py -- \
  --input /root/workspace/pacsim/urdf/Model_without_Steering_Tires.stl \
  --output-dir /root/workspace/pacsim/urdf/repaired_body_variants \
  --merge-distance 0.01 \
  --voxel-sizes 0.5,0.75,1.0,1.5,2.0

The script exports:
  - cleaned_no_remesh.stl for comparison/diagnostics
  - voxel_original_<size>.stl variants
  - voxel_cleaned_<size>.stl variants
  - car_body_repair_variants.blend for visual comparison

It intentionally does not overwrite the original STL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/root/workspace/pacsim/urdf/Model_without_Steering_Tires.stl",
        help="Input body STL.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/workspace/pacsim/urdf/repaired_body_variants",
        help="Directory for exported candidate meshes.",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=0.01,
        help="Merge-by-distance threshold in raw STL units.",
    )
    parser.add_argument(
        "--voxel-sizes",
        default="0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated voxel sizes in raw STL units.",
    )
    parser.add_argument(
        "--voxel-source",
        choices=("original", "cleaned", "both"),
        default="both",
        help=(
            "Which mesh to voxel-remesh. 'original' often works better for duplicate-sheet STL files; "
            "'cleaned' can make near-coplanar faces exactly coplanar."
        ),
    )
    parser.add_argument(
        "--decimate-ratio",
        type=float,
        default=1.0,
        help="Optional decimate ratio after voxel remesh. 1.0 disables decimation.",
    )
    return parser.parse_args(argv)


def select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.ops.object.mode_set.poll() else None
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def import_stl(path: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=path)
    else:
        bpy.ops.import_mesh.stl(filepath=path)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh object was imported from {path}")

    obj = mesh_objects[-1]
    obj.name = "car_body_original_import"
    select_only(obj)
    return obj


def export_stl(obj: bpy.types.Object, path: str) -> None:
    select_only(obj)
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    else:
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)


def clean_mesh(obj: bpy.types.Object, merge_distance: float) -> None:
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    try:
        bpy.ops.mesh.merge_by_distance(distance=merge_distance)
    except Exception:
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)

    try:
        bpy.ops.mesh.dissolve_degenerate(threshold=merge_distance * 0.1)
    except Exception:
        pass

    try:
        bpy.ops.mesh.delete_loose()
    except Exception:
        pass

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def add_weighted_normals(obj: bpy.types.Object) -> None:
    select_only(obj)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass

    mod = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    mod.keep_sharp = True
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass


def duplicate_object(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    select_only(obj)
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = name
    bpy.context.collection.objects.link(dup)
    select_only(dup)
    return dup


def voxel_remesh(obj: bpy.types.Object, voxel_size: float, decimate_ratio: float) -> None:
    select_only(obj)
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()

    if decimate_ratio < 0.999:
        mod = obj.modifiers.new("decimate_after_voxel", "DECIMATE")
        mod.ratio = max(0.01, min(1.0, decimate_ratio))
        bpy.ops.object.modifier_apply(modifier=mod.name)

    clean_mesh(obj, voxel_size * 0.01)
    add_weighted_normals(obj)


def mesh_stats(obj: bpy.types.Object) -> str:
    mesh = obj.data
    return f"verts={len(mesh.vertices)} edges={len(mesh.edges)} faces={len(mesh.polygons)}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voxel_sizes = [
        float(value.strip())
        for value in args.voxel_sizes.split(",")
        if value.strip()
    ]

    original = import_stl(str(input_path))
    cleaned = duplicate_object(original, "car_body_cleaned_no_remesh")
    original.hide_set(True)
    original.hide_render = True

    clean_mesh(cleaned, args.merge_distance)
    add_weighted_normals(cleaned)
    cleaned_path = output_dir / "cleaned_no_remesh.stl"
    export_stl(cleaned, str(cleaned_path))
    print(f"exported {cleaned_path} ({mesh_stats(cleaned)})")

    sources = []
    if args.voxel_source in ("original", "both"):
        sources.append(("original", original))
    if args.voxel_source in ("cleaned", "both"):
        sources.append(("cleaned", cleaned))

    for source_name, source_obj in sources:
        for voxel_size in voxel_sizes:
            candidate = duplicate_object(source_obj, f"car_body_voxel_{source_name}_{voxel_size:g}")
            candidate.hide_set(False)
            candidate.hide_render = False
            voxel_remesh(candidate, voxel_size, args.decimate_ratio)
            safe_size = str(voxel_size).replace(".", "p")
            candidate_path = output_dir / f"voxel_{source_name}_{safe_size}.stl"
            export_stl(candidate, str(candidate_path))
            print(f"exported {candidate_path} ({mesh_stats(candidate)})")

    blend_path = output_dir / "car_body_repair_variants.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"saved {blend_path}")


if __name__ == "__main__":
    main()
