"""Decimate a clean pacsim body STL back toward the original mesh size.

This is intended for the workflow:

1. Extract the clean GLB body in URDF STL units:

   python3 /root/workspace/pacsim/scripts/extract_glb_body_to_urdf_stl.py \
     --output /root/workspace/pacsim/urdf/Model_without_Steering_Tires_from_glb.stl

2. Run this script in Blender:

   blender --background --python /root/workspace/pacsim/scripts/blender_decimate_body_stl.py -- \
     --input /root/workspace/pacsim/urdf/Model_without_Steering_Tires_from_glb.stl \
     --output /root/workspace/pacsim/urdf/Model_without_Steering_Tires_fixed_lowpoly.stl \
     --target-faces 96000

The output remains in the same raw STL coordinate/scale convention as the
URDF body mesh, so separate_model.xacro should keep scale="0.01 0.01 0.01".
"""

from __future__ import annotations

import argparse
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
        default="/root/workspace/pacsim/urdf/Model_without_Steering_Tires_from_glb.stl",
        help="Input clean STL in URDF raw units.",
    )
    parser.add_argument(
        "--output",
        default="/root/workspace/pacsim/urdf/Model_without_Steering_Tires_fixed_lowpoly.stl",
        help="Output decimated STL.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=96000,
        help="Approximate target face count. Original body STL is about 96k triangles.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Optional explicit decimate ratio. Overrides --target-faces.",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=0.0,
        help="Optional merge-by-distance threshold before decimation.",
    )
    return parser.parse_args(argv)


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
    obj.name = "car_body_decimate_source"
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def export_stl(obj: bpy.types.Object, path: str) -> None:
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.ops.object.mode_set.poll() else None
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    else:
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)


def clean_normals(obj: bpy.types.Object, merge_distance: float) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    if merge_distance > 0.0:
        try:
            bpy.ops.mesh.merge_by_distance(distance=merge_distance)
        except Exception:
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)

    try:
        bpy.ops.mesh.dissolve_degenerate(threshold=max(merge_distance * 0.1, 1e-7))
    except Exception:
        pass

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def decimate(obj: bpy.types.Object, target_faces: int, ratio_override: float | None) -> float:
    current_faces = max(1, len(obj.data.polygons))
    ratio = ratio_override if ratio_override is not None else float(target_faces) / float(current_faces)
    ratio = max(0.01, min(1.0, ratio))

    mod = obj.modifiers.new("decimate_to_target", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    try:
        mod.use_collapse_triangulate = True
    except Exception:
        pass
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return ratio


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obj = import_stl(str(input_path))
    clean_normals(obj, args.merge_distance)
    before = len(obj.data.polygons)
    ratio = decimate(obj, args.target_faces, args.ratio)
    clean_normals(obj, 0.0)
    after = len(obj.data.polygons)
    export_stl(obj, str(output_path))

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"faces_before={before}")
    print(f"faces_after={after}")
    print(f"decimate_ratio={ratio:.6g}")
    print('keep separate_model.xacro scale="0.01 0.01 0.01"')


if __name__ == "__main__":
    main()
