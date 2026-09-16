#!/usr/bin/env python3
"""Build and stage a reproducible SwiftShader runtime artifact for packaging."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPOSITORY = "https://swiftshader.googlesource.com/SwiftShader"
DEFAULT_REVISION = "89556131bf9d48af3c5c9fbb9a3322e706da89a3"

RUNTIME_LIBRARY_NAMES = {
    "Linux": ["libvk_swiftshader.so"],
    "Darwin": ["libvk_swiftshader.dylib"],
    "Windows": ["vk_swiftshader.dll", "libvk_swiftshader.dll"],
}

DIRECT_LOADER_ALIASES = {
    "Linux": ["libvulkan.so.1", "libvulkan.so"],
    "Darwin": ["libvulkan.1.dylib", "libvulkan.dylib"],
    "Windows": ["vulkan-1.dll"],
}

NOTICE_FILES = [
    "LICENSE.txt",
    "AUTHORS.txt",
    "CONTRIBUTORS.txt",
]


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def git_command(*arguments: str) -> list[str]:
    """Use HTTP/1.1 for reliable GitHub submodule fetches behind proxies."""
    return ["git", "-c", "http.version=HTTP/1.1", *arguments]


def ensure_source(repository: str, revision: str, source_dir: Path) -> str:
    if source_dir.exists() and not (source_dir / ".git").is_dir():
        raise RuntimeError(f"{source_dir} exists but is not a git checkout")
    if not source_dir.exists():
        run(git_command("clone", "--depth", "1", "--no-checkout", repository, str(source_dir)))

    try:
        run(git_command("fetch", "--depth", "1", "origin", revision), cwd=source_dir)
    except subprocess.CalledProcessError:
        run(git_command("fetch", "--depth", "1", "origin"), cwd=source_dir)

    run(git_command("checkout", "--detach", revision), cwd=source_dir)
    run(
        git_command("submodule", "update", "--init", "--recursive", "--depth", "1"),
        cwd=source_dir,
    )
    return command_output(git_command("rev-parse", "HEAD"), cwd=source_dir)


def find_file(root: Path, names: list[str]) -> Path:
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name in names and not path.name.endswith(".tmp")
    ]
    if not matches:
        raise RuntimeError(f"Could not find any of {names} below {root}")
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def stage_icd_json(source: Path, destination: Path, library_path: str) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    icd = data.setdefault("ICD", {})
    icd["library_path"] = library_path
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def stage_runtime(
    source_dir: Path,
    build_dir: Path,
    prefix: Path,
    *,
    direct_loader_alias: bool,
    requested_revision: str,
    actual_revision: str,
    repository: str,
    config: str,
    cmake_args: list[str],
    absolute_library_path: bool,
) -> None:
    system_name = platform.system()
    library_names = RUNTIME_LIBRARY_NAMES.get(system_name)
    if library_names is None:
        supported = ", ".join(sorted(RUNTIME_LIBRARY_NAMES))
        raise RuntimeError(f"Unsupported platform {system_name!r}; supported platforms: {supported}")

    prefix.mkdir(parents=True, exist_ok=True)

    icd_json = find_file(build_dir, ["vk_swiftshader_icd.json"])
    runtime_library = find_file(build_dir, library_names)

    staged_library = prefix / runtime_library.name
    shutil.copy2(runtime_library, staged_library)
    icd_library_path = str(staged_library) if absolute_library_path else staged_library.name
    stage_icd_json(icd_json, prefix / "vk_swiftshader_icd.json", icd_library_path)

    staged_files = [staged_library.name, "vk_swiftshader_icd.json"]
    if direct_loader_alias:
        for alias in DIRECT_LOADER_ALIASES[system_name]:
            shutil.copy2(runtime_library, prefix / alias)
            staged_files.append(alias)

    for name in NOTICE_FILES:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, prefix / name)
            staged_files.append(name)

    manifest = {
        "repository": repository,
        "requested_revision": requested_revision,
        "revision": actual_revision,
        "platform": system_name,
        "machine": platform.machine(),
        "config": config,
        "cmake_args": cmake_args,
        "runtime_files": sorted(staged_files),
    }
    (prefix / "swiftshader_build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    default_work_dir = Path.home() / ".cache" / "pacsim" / "swiftshader"
    parser = argparse.ArgumentParser(
        description="Build SwiftShader from a pinned upstream revision and stage the Vulkan ICD runtime."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-dir", type=Path, default=default_work_dir / "src")
    parser.add_argument("--build-dir", type=Path, default=default_work_dir / "build")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--config", default="Release")
    parser.add_argument("--target", default="vk_swiftshader")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--generator")
    parser.add_argument(
        "--direct-loader-alias",
        action="store_true",
        help="Also copy the SwiftShader ICD library under the platform Vulkan loader name.",
    )
    parser.add_argument(
        "--absolute-library-path",
        action="store_true",
        help="Write an absolute ICD library_path pointing at the staged runtime library.",
    )
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="Additional argument passed to the SwiftShader CMake configure step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    prefix = args.prefix.resolve()

    actual_revision = ensure_source(args.repository, args.revision, source_dir)

    configure_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={args.config}",
    ]
    if args.generator:
        configure_command.extend(["-G", args.generator])
    configure_command.extend(args.cmake_arg)
    run(configure_command)

    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            args.config,
            "--target",
            args.target,
            "--parallel",
            str(args.jobs),
        ]
    )

    stage_runtime(
        source_dir,
        build_dir,
        prefix,
        direct_loader_alias=args.direct_loader_alias,
        requested_revision=args.revision,
        actual_revision=actual_revision,
        repository=args.repository,
        config=args.config,
        cmake_args=args.cmake_arg,
        absolute_library_path=args.absolute_library_path,
    )

    print(f"SwiftShader runtime staged in: {prefix}")
    print(f"ICD JSON: {prefix / 'vk_swiftshader_icd.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
