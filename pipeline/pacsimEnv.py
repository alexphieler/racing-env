import os
import sys
import ctypes
import importlib
import json
import tempfile
import time

import gymnasium as gym
import numpy as np
import termcolor
from gymnasium import spaces
from shapely import contains_xy, prepare
from shapely.geometry.polygon import Polygon

# Prefer install/package module paths and package build dirs.
# Avoid top-level build dirs that can contain stale pacsim_pybind*.so artifacts.
_PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
_PIPELINE_CAMERA_CONFIG = os.path.join(_MODULE_DIR, "cameras.yaml")
_PACSIM_WS = os.path.join(_WORKSPACE_ROOT, "pacsim_ws")
_IMAGE_PACSIM_WS = "/root/workspace/pacsim_ws"


def _unique_existing_or_candidate_paths(paths):
    ret = []
    seen = set()
    for path in paths:
        if(path is None or str(path).strip() == ""):
            continue
        path = os.path.abspath(os.path.expanduser(str(path)))
        if(path not in seen):
            seen.add(path)
            ret.append(path)
    return ret


_PACSIM_BINARY_WS_CANDIDATES = _unique_existing_or_candidate_paths([
    os.environ.get("PACSIM_WS"),
    _IMAGE_PACSIM_WS,
    os.path.join(_WORKSPACE_ROOT, "pacsim_ws_swiftshader"),
    _PACSIM_WS,
])
_PACSIM_PYBIND_CANDIDATES = []
_PACSIM_LIB_CANDIDATES = []
_PACSIM_SWIFTSHADER_ICD_CANDIDATES = [
    "/opt/pacsim-swiftshader/vk_swiftshader_icd.json",
]
for _PACSIM_BINARY_WS in _PACSIM_BINARY_WS_CANDIDATES:
    _PACSIM_PYBIND_CANDIDATES.extend([
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "pacsim"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "lib", f"python{_PY_VER}", "site-packages"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim_reconf", "lib", f"python{_PY_VER}", "site-packages"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim", "pacsim"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim_reconf"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim_reconf", "pacsim"),
    ])
    _PACSIM_LIB_CANDIDATES.extend([
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "lib", "pacsim", "libpacsim_lib.so"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim_reconf", "lib", "pacsim", "libpacsim_lib.so"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim", "libpacsim_lib.so"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim", "pacsim", "libpacsim_lib.so"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim_reconf", "libpacsim_lib.so"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim_reconf", "pacsim", "libpacsim_lib.so"),
    ])
    _PACSIM_SWIFTSHADER_ICD_CANDIDATES.extend([
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "pacsim", "swiftshader", "vk_swiftshader_icd.json"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "lib", "pacsim", "swiftshader", "vk_swiftshader_icd.json"),
        os.path.join(_PACSIM_BINARY_WS, "install", "pacsim", "lib", f"python{_PY_VER}", "site-packages", "swiftshader", "vk_swiftshader_icd.json"),
        os.path.join(_PACSIM_BINARY_WS, "build", "pacsim", "swiftshader", "vk_swiftshader_icd.json"),
    ])


def _prepend_existing_paths(paths):
    for path in reversed(paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _clear_failed_pacsim_imports():
    for name in list(sys.modules):
        if(name == "pacsim_pybind" or name == "pacsim" or name.startswith("pacsim.")):
            sys.modules.pop(name, None)


def _import_pacsim_pybind():
    return importlib.import_module("pacsim_pybind")


def _import_pacsim_pybind_from_candidates(paths):
    errors = []
    for path in paths:
        if(not os.path.isdir(path)):
            continue

        original_sys_path = list(sys.path)
        sys.path[:] = [path] + [
            existing for existing in sys.path
            if(os.path.abspath(os.path.expanduser(str(existing))) != path)
        ]
        _clear_failed_pacsim_imports()
        try:
            return _import_pacsim_pybind(), errors
        except (ImportError, OSError) as exc:
            errors.append("{0}: {1}".format(path, exc))
            sys.path[:] = original_sys_path

    return None, errors


def _load_first_existing_shared_lib(paths):
    errors = []
    for path in paths:
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return path, errors
            except OSError as exc:
                errors.append("{0}: {1}".format(path, exc))
    return None, errors


def _env_path_has_existing_file(value):
    if(value is None or str(value).strip() == ""):
        return False
    return any(os.path.isfile(os.path.expanduser(path)) for path in str(value).split(os.pathsep) if path)


def _truthy_env_var(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _absolute_swiftshader_icd(path):
    path = os.path.abspath(os.path.expanduser(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        icd = data.get("ICD")
        if(not isinstance(icd, dict)):
            return path
        library_path = icd.get("library_path")
        if(not isinstance(library_path, str) or library_path.strip() == ""):
            return path
        if(os.path.isabs(library_path) and os.path.isfile(library_path)):
            return path

        library_abs = os.path.abspath(os.path.join(os.path.dirname(path), library_path))
        if(not os.path.isfile(library_abs)):
            return path

        icd["library_path"] = library_abs
        destination = os.path.join(
            tempfile.gettempdir(),
            "pacsim_swiftshader_icd_{0}.json".format(abs(hash((path, library_abs))))
        )
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        return destination
    except (OSError, ValueError, TypeError):
        return path


def _set_swiftshader_icd(path):
    path = _absolute_swiftshader_icd(path)
    os.environ["PACSIM_SWIFTSHADER_ICD"] = path
    if(_truthy_env_var("PACSIM_FORCE_SWIFTSHADER")):
        os.environ["VK_ICD_FILENAMES"] = path
    return path


def _configure_swiftshader_runtime():
    explicit = os.environ.get("PACSIM_SWIFTSHADER_ICD")
    if(_env_path_has_existing_file(explicit)):
        return _set_swiftshader_icd(explicit)

    for path in _PACSIM_SWIFTSHADER_ICD_CANDIDATES:
        if(os.path.isfile(path)):
            return _set_swiftshader_icd(path)
    return None


def _configure_swiftshader_for_imported_pacsim():
    explicit = os.environ.get("PACSIM_SWIFTSHADER_ICD")
    if(_env_path_has_existing_file(explicit)):
        return _set_swiftshader_icd(explicit)

    candidates = []
    module_file = getattr(pacsim_pybind, "__file__", None)
    if(module_file):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(module_file)), "swiftshader", "vk_swiftshader_icd.json"))

    try:
        pacsim_package = importlib.import_module("pacsim")
        package_file = getattr(pacsim_package, "__file__", None)
        if(package_file):
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(package_file)), "swiftshader", "vk_swiftshader_icd.json"))
        bundled_path = None
        bundled_path_getter = getattr(pacsim_package, "bundled_swiftshader_icd_path", None)
        if(callable(bundled_path_getter)):
            bundled_path = bundled_path_getter()
        if(_env_path_has_existing_file(bundled_path)):
            return _set_swiftshader_icd(bundled_path)
    except (ImportError, OSError, AttributeError):
        pass

    for path in _unique_existing_or_candidate_paths(candidates):
        if(os.path.isfile(path)):
            return _set_swiftshader_icd(path)
    return _configure_swiftshader_runtime()


try:
    import yaml
except ImportError:
    yaml = None

pacsim_pybind = None
_pacsim_pybind_import_errors = []
try:
    pacsim_pybind = _import_pacsim_pybind()
except (ImportError, OSError) as exc:
    _pacsim_pybind_import_errors.append("default sys.path: {0}".format(exc))

if(pacsim_pybind is None):
    _pacsim_pybind_search_paths = _unique_existing_or_candidate_paths(_PACSIM_PYBIND_CANDIDATES + sys.path)
    pacsim_pybind, _candidate_import_errors = _import_pacsim_pybind_from_candidates(_pacsim_pybind_search_paths)
    _pacsim_pybind_import_errors.extend(_candidate_import_errors)

if(pacsim_pybind is None):
    _loaded_pacsim_lib, _pacsim_lib_load_errors = _load_first_existing_shared_lib(_PACSIM_LIB_CANDIDATES)
    if(_loaded_pacsim_lib is not None):
        _clear_failed_pacsim_imports()
        try:
            pacsim_pybind = _import_pacsim_pybind()
        except (ImportError, OSError) as second_exc:
            raise ImportError(
                "Could not import pacsim_pybind after loading {0}. Rebuild pacsim as standalone "
                "with `-DPACSIM_BUILD_ROS=OFF -DPACSIM_BUILD_PYTHON=ON` for the headless image."
                .format(_loaded_pacsim_lib)
            ) from second_exc
    else:
        detail_parts = []
        if(_pacsim_pybind_import_errors):
            detail_parts.append("Python import attempts failed: " + " | ".join(_pacsim_pybind_import_errors))
        if(_pacsim_lib_load_errors):
            detail_parts.append("Native library load attempts failed: " + " | ".join(_pacsim_lib_load_errors))
        detail = ""
        if(detail_parts):
            detail = " " + " ".join(detail_parts)
        raise ImportError(
            "Could not import pacsim_pybind. Build pacsim_ws first with "
            "`-DPACSIM_BUILD_ROS=OFF -DPACSIM_BUILD_PYTHON=ON` for the headless image."
            + detail
        )

_configure_swiftshader_for_imported_pacsim()

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

class pacsimEnv(gym.Env):
    def _default_map_files(self):
        track_directory = os.path.join(_MODULE_DIR, "generated_tracks")
        if(not os.path.isdir(track_directory)):
            raise FileNotFoundError(
                "Random-track directory does not exist: {0}".format(track_directory)
            )

        map_files = sorted(
            os.path.join(track_directory, file_name)
            for file_name in os.listdir(track_directory)
            if(
                os.path.isfile(os.path.join(track_directory, file_name))
                and file_name.lower().endswith((".yaml", ".yml"))
            )
        )
        if(not map_files):
            raise ValueError(
                "Random-track directory contains no YAML track files: {0}".format(track_directory)
            )
        return map_files

    def _camera_obs_key_base(self, camera_name):
        lower = str(camera_name).strip().lower()
        if(lower == "camera_left" or lower == "left"):
            return "cameraLeft"
        if(lower == "camera_front" or lower == "front"):
            return "cameraFront"
        if(lower == "camera_right" or lower == "right"):
            return "cameraRight"

        sanitized = ""
        for c in str(camera_name):
            if(c.isalnum()):
                sanitized += c
            else:
                sanitized += "_"
        sanitized = sanitized.strip("_")
        if(sanitized == ""):
            sanitized = "camera"
        return "camera_" + sanitized

    def _build_camera_obs_keys(self, camera_names):
        ret = []
        used = set()
        for name in camera_names:
            base = self._camera_obs_key_base(name)
            key = base
            suffix = 1
            while(key in used):
                suffix += 1
                key = base + "_" + str(suffix)
            used.add(key)
            ret.append(key)
        return ret

    def _resolve_camera_config_path(self, requested_path):
        # Respect explicit user input first.
        if(requested_path is not None and str(requested_path).strip() != ""):
            requested = os.path.expanduser(str(requested_path))
            if(os.path.isabs(requested)):
                explicit_candidates = [requested]
            else:
                explicit_candidates = [
                    os.path.abspath(requested),
                    os.path.join(_MODULE_DIR, requested),
                ]

            checked = _unique_existing_or_candidate_paths(explicit_candidates)
            for candidate in checked:
                if(os.path.isfile(candidate)):
                    return candidate
            raise RuntimeError("Camera config file not found. Checked: {0}".format(", ".join(checked)))

        # Auto-discover cameras.yaml from common workspace/install layouts.
        checked = []
        checked_set = set()

        def add_candidate(path):
            normalized = os.path.abspath(path)
            if(normalized not in checked_set):
                checked_set.add(normalized)
                checked.append(normalized)

        seed_dirs = [
            os.path.dirname(os.path.abspath(__file__)),
            os.getcwd(),
        ]
        try:
            seed_dirs.append(os.path.dirname(os.path.abspath(pacsim_pybind.__file__)))
        except:
            pass
        for entry in sys.path:
            if(isinstance(entry, str) and len(entry) > 0):
                seed_dirs.append(entry)

        rel_candidates = [
            "cameras.yaml",
            "cameraSensors.yaml",
            os.path.join("src", "config", "cameras.yaml"),
            os.path.join("config", "pacsim", "cameras.yaml"),
            os.path.join("config", "cameras.yaml"),
            os.path.join("src", "config", "cameraSensors.yaml"),
            os.path.join("config", "pacsim", "cameraSensors.yaml"),
            os.path.join("config", "cameraSensors.yaml"),
        ]

        add_candidate(_PIPELINE_CAMERA_CONFIG)
        for rel in rel_candidates:
            add_candidate(os.path.join(_PACSIM_WS, rel))

        for seed in seed_dirs:
            current = os.path.abspath(seed)
            for _ in range(7):
                for rel in rel_candidates:
                    add_candidate(os.path.join(current, rel))
                parent = os.path.dirname(current)
                if(parent == current):
                    break
                current = parent

        for candidate in checked:
            if(os.path.isfile(candidate)):
                return candidate

        raise RuntimeError(
            "camera_config_file must be set when using pacsimEnv. "
            "Could not auto-discover cameras.yaml from workspace/install paths."
        )

    def _resolve_existing_path(self, requested_path, default_path, label):
        if(requested_path is not None and str(requested_path).strip() != ""):
            candidate = os.path.abspath(os.path.expanduser(str(requested_path)))
        else:
            candidate = os.path.abspath(default_path)
        if(os.path.isfile(candidate)):
            return candidate
        raise RuntimeError("{0} file not found: {1}".format(label, candidate))

    def _resolve_pacsim_source_root(self):
        candidates = [
            os.path.join(_PACSIM_WS, "src", "pacsim"),
            os.path.join(_PACSIM_WS, "src"),
            os.path.join(_WORKSPACE_ROOT, "pacsim"),
        ]
        required_files = [
            os.path.join("urdf", "separate_model.xacro"),
            os.path.join("config", "vehicleModel.yaml"),
        ]
        for candidate in candidates:
            root = os.path.abspath(candidate)
            if(all(os.path.isfile(os.path.join(root, rel)) for rel in required_files)):
                return root
        raise RuntimeError(
            "Could not find pacsim source root. Checked: {0}".format(
                ", ".join(os.path.abspath(c) for c in candidates)
            )
        )

    def _load_camera_metadata(self, config_path):
        if(config_path is None):
            raise RuntimeError("camera_config_file must be set when using pacsimEnv.")
        if(not os.path.isfile(config_path)):
            raise RuntimeError("Camera config file not found: {0}".format(config_path))
        if(yaml is None):
            raise RuntimeError(
                "PyYAML is required for camera_config_file support. Install with: pip install pyyaml"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if(cfg is None):
            raise RuntimeError("Camera config file is empty: {0}".format(config_path))
        if(not isinstance(cfg, dict)):
            raise RuntimeError("Camera config must be a YAML mapping.")

        camera_entries = cfg.get("cameras", [])
        if(not isinstance(camera_entries, list)):
            raise RuntimeError("Camera config key 'cameras' must be a list.")

        names = []
        enabled = []
        rate_hz = 10.0
        delay_mean = 0.0
        resolution = None

        for item in camera_entries:
            if(not isinstance(item, dict)):
                continue

            sensor_cfg = item.get("sensor", item)
            if(not isinstance(sensor_cfg, dict)):
                sensor_cfg = {}

            # Support both formats:
            # - sensor: { ... }
            # - sensor:\n    name: ... (name/rate/etc at same level as sensor)
            merged_cfg = dict(sensor_cfg)
            for k, v in item.items():
                if(k != "sensor" and k not in merged_cfg):
                    merged_cfg[k] = v

            is_enabled = bool(merged_cfg.get("enabled", True))
            name = str(merged_cfg.get("name", "camera_{0}".format(len(names))))
            names.append(name)
            enabled.append(is_enabled)

            if("rate" in merged_cfg):
                try:
                    rate_val = float(merged_cfg["rate"])
                    if(rate_val > 0.0):
                        rate_hz = rate_val
                except:
                    pass

            delay_cfg = merged_cfg.get("delay", {})
            if(isinstance(delay_cfg, dict) and ("mean" in delay_cfg)):
                try:
                    delay_mean = float(delay_cfg["mean"])
                except:
                    pass

            res_cfg = merged_cfg.get("resolution", {})
            if(isinstance(res_cfg, dict) and ("x" in res_cfg) and ("y" in res_cfg)):
                cam_w = int(res_cfg["x"])
                cam_h = int(res_cfg["y"])
                if(resolution is None):
                    resolution = (cam_w, cam_h)
                elif(resolution[0] != cam_w or resolution[1] != cam_h):
                    raise RuntimeError(
                        "Camera config currently requires identical resolution for all cameras."
                    )

        if(len(names) == 0):
            raise RuntimeError("Camera config must define at least one camera entry.")
        if(resolution is None):
            raise RuntimeError("Camera config must define resolution.x and resolution.y for cameras.")

        return {
            "names": names,
            "enabled": enabled,
            "rate_hz": rate_hz,
            "delay_mean": delay_mean,
            "resolution": resolution,
        }

    def __init__(self, param=None):
        if(param is None):
            param = {}

        try:
            self.printStepStatus = param["print_step_status"]
        except:
            self.printStepStatus = True
        # Keep the historical default (messages are enabled), while allowing
        # high-throughput callers such as vectorized benchmarks to avoid
        # terminal I/O in their timed path.
        self.verbose = bool(param.get("verbose", True))

        try:
            self.profileEnvStep = param["profile_env_step"]
        except:
            self.profileEnvStep = False
        self.profileEnvStepTotals = {}
        self.profileEnvStepCounts = {}

        if(self.printStepStatus and self.verbose):
            print("Init, params {0}".format(param))
        self.useCamSim = False
        try:
            self.useCamSim = param["cam_sim"]
        except:
            self.useCamSim = True
        try:
            self.includeCameraObs = bool(param["include_camera_obs"])
        except:
            self.includeCameraObs = True
        if(self.useCamSim):
            self.includeCameraObs = True
        self.cameraRenderer = str(param.get("camera_renderer", param.get("camera_backend", "vulkan"))).strip().lower()
        if(self.cameraRenderer in ("vk",)):
            self.cameraRenderer = "vulkan"
        if(self.cameraRenderer != "vulkan"):
            raise RuntimeError("camera_renderer must be 'vulkan', got '{0}'.".format(self.cameraRenderer))

        if(bool(param.get("compare_camera_renderers", False))):
            raise RuntimeError("compare_camera_renderers is no longer supported; pacsim is Vulkan-only.")

        try:
            self.lambda_progress = param["lambda_progress"]
        except:
            self.lambda_progress = 0.02
        
        try:
            self.lambda_tracking = param["lambda_tracking"]
        except:
            self.lambda_tracking = 0.003

        try:
            self.lambda_finish = param["lambda_finish"]
        except:
            self.lambda_finish = 10.0

        try:
            self.lambda_collition = param["lambda_collition"]
        except:
            self.lambda_collition = 10.0

        try:
            self.lambda_stand = param["lambda_stand"]
        except:
            self.lambda_stand = 0.5

        try:
            self.lambda_slipAngle = param["lambda_slipAngle"]
        except:
            self.lambda_slipAngle = 0.005

        try:
            self.lambda_slipRatio = param["lambda_slipRatio"]
        except:
            self.lambda_slipRatio = 0.05

        try:
            self.lambda_actionRate = param["lambda_actionRate"]
        except:
            self.lambda_actionRate = 0.002

        try:
            self.lambda_lateral_consistency = param["lambda_lateral_consistency"]
        except:
            self.lambda_lateral_consistency = 0.001

        try:
            self.lambda_longitudinal_consistency = param["lambda_longitudinal_consistency"]
        except:
            self.lambda_longitudinal_consistency = 0.0002

        try:
            camera_config_path = param["camera_config_file"]
        except:
            camera_config_path = None

        camera_config_path = self._resolve_camera_config_path(camera_config_path)

        loaded_cam_cfg = self._load_camera_metadata(camera_config_path)
        self.cameraNames = [str(c) for c in loaded_cam_cfg["names"]]
        self.cameraEnabled = [bool(c) for c in loaded_cam_cfg["enabled"]]
        self.cameraObsKeys = self._build_camera_obs_keys(self.cameraNames)
        self.cameraRateHz = loaded_cam_cfg["rate_hz"]
        self.cameraDelayMean = loaded_cam_cfg["delay_mean"]
        self.cameraWidth, self.cameraHeight = loaded_cam_cfg["resolution"]

        self.modelRoot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
        self.pacsimSourceRoot = self._resolve_pacsim_source_root()
        try:
            car_xacro_path = param["car_xacro_path"]
        except:
            car_xacro_path = None
        try:
            left_cone_asset_path = param["left_cone_asset_path"]
        except:
            left_cone_asset_path = None
        try:
            right_cone_asset_path = param["right_cone_asset_path"]
        except:
            right_cone_asset_path = None
        try:
            ground_plane_asset_path = param["ground_plane_asset_path"]
        except:
            ground_plane_asset_path = None
        try:
            skybox_asset_path = param["skybox_asset_path"]
        except:
            skybox_asset_path = None
        self.carXacroPath = self._resolve_existing_path(
            car_xacro_path,
            os.path.join(self.pacsimSourceRoot, "urdf", "separate_model.xacro"),
            "car_xacro_path",
        )
        self.leftConeAssetPath = self._resolve_existing_path(
            left_cone_asset_path,
            os.path.join(self.pacsimSourceRoot, "assets", "cones", "blue.glb"),
            "left_cone_asset_path",
        )
        self.rightConeAssetPath = self._resolve_existing_path(
            right_cone_asset_path,
            os.path.join(self.pacsimSourceRoot, "assets", "cones", "yellow.glb"),
            "right_cone_asset_path",
        )
        self.groundPlaneAssetPath = self._resolve_existing_path(
            ground_plane_asset_path,
            os.path.join(self.pacsimSourceRoot, "assets", "ground", "paving.glb"),
            "ground_plane_asset_path",
        )
        self.skyboxAssetPath = self._resolve_existing_path(
            skybox_asset_path,
            os.path.join(self.pacsimSourceRoot, "assets", "sky", "skysphere.glb"),
            "skybox_asset_path",
        )
        try:
            self.asyncReadback = param["async_readback"]
        except:
            self.asyncReadback = False
        try:
            self.useShadows = param["use_shadows"]
        except:
            self.useShadows = True
        if(self.useCamSim):
            self.cppCameraSim = self._create_cpp_camera_sim(self.cameraRenderer, camera_config_path)
            self._configure_cpp_camera_sim(self.cppCameraSim)

            try:
                if(hasattr(self.cppCameraSim, "width") and hasattr(self.cppCameraSim, "height")):
                    self.cameraWidth = int(self.cppCameraSim.width())
                    self.cameraHeight = int(self.cppCameraSim.height())
                if(hasattr(self.cppCameraSim, "cameraNames")):
                    names_from_cpp = list(self.cppCameraSim.cameraNames())
                    if(len(names_from_cpp) > 0):
                        self.cameraNames = [str(n) for n in names_from_cpp]
                        self.cameraObsKeys = self._build_camera_obs_keys(self.cameraNames)
                        self.cameraEnabled = [True for _ in self.cameraNames]
                if(hasattr(self.cppCameraSim, "cameraEnabledFlags")):
                    enabled_from_cpp = list(self.cppCameraSim.cameraEnabledFlags())
                    if(len(enabled_from_cpp) == len(self.cameraNames)):
                        self.cameraEnabled = [bool(v) for v in enabled_from_cpp]
                if(hasattr(self.cppCameraSim, "cameraRatesHz")):
                    rates_from_cpp = [float(v) for v in list(self.cppCameraSim.cameraRatesHz())]
                    positive_rates = [r for r in rates_from_cpp if r > 0.0]
                    if(len(positive_rates) > 0):
                        self.cameraRateHz = positive_rates[0]
                if(hasattr(self.cppCameraSim, "cameraDelayMeans")):
                    delays_from_cpp = [float(v) for v in list(self.cppCameraSim.cameraDelayMeans())]
                    if(len(delays_from_cpp) > 0):
                        self.cameraDelayMean = delays_from_cpp[0]
            except Exception as e:
                raise RuntimeError("Failed to load C++ camera config '{0}': {1}".format(camera_config_path, e))

        # Lightweight Python-side timing for the C++ renderer call.
        try:
            self.profileRender = param["profile_render"]
        except:
            self.profileRender = False
        self.profileRenderEvery = 60
        self.profileRenderSamples = 0
        self.profileRenderMsSum = 0.0
        self.profileRenderMsMin = float("inf")
        self.profileRenderMsMax = 0.0
        self.M = 8
        self.rangefinder_angles = np.linspace(-1.0, 1.0, 2*self.M+1)
        self.rangefinder_angles = np.power(np.abs(self.rangefinder_angles),1) * np.sign(self.rangefinder_angles)
        self.rangefinder_angles = self.rangefinder_angles * np.pi/2.0
        self.defaultDeadTime = float(param.get("dead_time", 0.05))
        self.defaultSteeringDeadTime = float(param.get(
            "steering_dead_time",
            param.get("dead_time_steering", self.defaultDeadTime),
        ))
        self.defaultTorqueDeadTime = float(param.get(
            "torque_dead_time",
            param.get("dead_time_torque", self.defaultDeadTime),
        ))

        self.interval = 1.0/self.cameraRateHz
        self.pFL = np.array([1.65,0.72+0.11,0.0])
        self.pFR = np.array([1.65,-(0.72+0.11),0.0])
        self.pRL = np.array([-1.0,0.72+0.11,0.0])
        self.pRR = np.array([-1.0,-(0.72+0.11),0.0])
        self.carCornerOffsets = np.array([
            self.pFL[0:2],
            self.pFR[0:2],
            self.pRL[0:2],
            self.pRR[0:2],
        ])
        self.maxStartPlacementAttempts = max(1, int(param.get("max_start_placement_attempts", 50)))

        self.useComplexModel = True
        self.outputRPM = True
        self.outputCurrentSteering = True
        self.outputLastAction = bool(param.get(
            "include_last_action",
            param.get("output_last_action", param.get("output_last_actions", True)),
        ))
        self.outputLastActions = self.outputLastAction
        self.outputIMU = True
        self.outputRangefinder = False

        self.minTorque = -22.0
        self.maxTorque = 22.0

        self.minSteering = -2.0
        self.maxSteering = 2.0
        
        self.maxRange = 75.0

        self.maxRpm = 20000.0

        self.maxSpeed = 40.0
        self.maxYawRate = 10.0

        self.maxImuAcceleration = 20.0
        
        self.emptyImageCode = None
        if(self.includeCameraObs):
            self.emptyImageCode = np.zeros((3, self.cameraWidth, self.cameraHeight), dtype=np.uint8)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        rangeLower = np.zeros((self.M*2+1), dtype=np.float32)
        rangeUpper = np.ones((self.M*2+1), dtype=np.float32)
        lowerBound = np.concatenate((rangeLower, np.array([-1.0, -1.0, -1.0], dtype=np.float32)))
        upperBound = np.concatenate((rangeUpper, np.array([1.0, 1.0, 1.0], dtype=np.float32)))
        self.outputDim = self.M*2+1+3
        velLower = np.array([-1.0, -1.0], dtype=np.float32)
        velUpper = np.array([1.0, 1.0], dtype=np.float32)
        rpmLower = None
        rpmUpper = None
        steerLower = None
        steerUpper = None
        imuLower = None
        imuUpper = None
        if(self.outputRPM):
            rpmLower = np.full((4,), -1000.0 / self.maxRpm, dtype=np.float32)
            rpmUpper = np.full((4,), 21_000.0 / self.maxRpm, dtype=np.float32)
            lowerBound = np.concatenate((lowerBound,rpmLower))
            upperBound = np.concatenate((upperBound,rpmUpper))
            self.outputDim += 4
        if(self.outputCurrentSteering):
            steerLower = np.full((1,), -1.0, dtype=np.float32)
            steerUpper = np.full((1,), 1.0, dtype=np.float32)
            lowerBound = np.concatenate((lowerBound,steerLower))
            upperBound = np.concatenate((upperBound,steerUpper))
            self.outputDim += 1
        if(self.outputLastActions):
            lastActionLower = np.full((5,), -1.0, dtype=np.float32)
            lastActionUpper = np.full((5,), 1.0, dtype=np.float32)
            lowerBound = np.concatenate((lowerBound,lastActionLower))
            upperBound = np.concatenate((upperBound,lastActionUpper))
            self.outputDim += 5
        if(self.outputIMU):
            imuLower = np.full((3,), -1.0, dtype=np.float32)
            imuUpper = np.full((3,), 1.0, dtype=np.float32)
            lowerBound = np.concatenate((lowerBound,imuLower))
            upperBound = np.concatenate((upperBound,imuUpper))
            self.outputDim += 3
        obsSpace = {
        "ranges": spaces.Box(low=rangeLower, high=rangeUpper, shape=(self.M*2+1,), dtype=np.float32),
        "velocity": spaces.Box(low=velLower, high=velUpper, shape=(2,), dtype=np.float32),
        "rpm": spaces.Box(low=rpmLower, high=rpmUpper, shape=(4,), dtype=np.float32),
        "steer": spaces.Box(low=steerLower, high=steerUpper, shape=(1,), dtype=np.float32),
        "imu": spaces.Box(low=imuLower, high=imuUpper, shape=(3,), dtype=np.float32),
        }
        if(self.outputLastActions):
            obsSpace["last_action"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32,
            )
        if(self.includeCameraObs):
            for camKey in self.cameraObsKeys:
                obsSpace[camKey] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.cameraWidth, self.cameraHeight,),
                    dtype=np.uint8,
                )
        self.observation_space = spaces.Dict(obsSpace)
        self.trackNr = 0
        self.defaultMapFiles = self._default_map_files()
        self.trackCache = {}
        self._preload_track_cache(self.defaultMapFiles)

        self.reset()

    def _create_cpp_camera_sim(self, renderer, camera_config_path):
        if(renderer != "vulkan"):
            raise RuntimeError("Only the Vulkan camera renderer is available.")
        if(not hasattr(pacsim_pybind, "VulkanCameraSim")):
            raise RuntimeError(
                "pacsim_pybind.VulkanCameraSim is missing. Rebuild pacsim_pybind in pacsim_ws first."
            )
        return pacsim_pybind.VulkanCameraSim(
            0,
            0,
            self.modelRoot,
            camera_config_path,
            self.carXacroPath,
        )

    def _configure_cpp_camera_sim(self, camera_sim):
        camera_sim.setAssetPaths(
            self.leftConeAssetPath,
            self.rightConeAssetPath,
            self.groundPlaneAssetPath,
            self.skyboxAssetPath,
        )
        camera_sim.setShadowsEnabled(bool(self.useShadows))

    def _profile_env_step_add(self, name, elapsed):
        if(not self.profileEnvStep):
            return
        self.profileEnvStepTotals[name] = self.profileEnvStepTotals.get(name, 0.0) + elapsed
        self.profileEnvStepCounts[name] = self.profileEnvStepCounts.get(name, 0) + 1

    def _build_track_cache_entry(self, mapFile, flipY):
        start_position = np.array([0.0, 0.0, 0.0])
        start_orientation = np.array([0.0, 0.0, 0.0])
        loaded_map = pacsim_pybind.loadMap(mapFile, start_position, start_orientation, flipY)

        left_lane = np.asarray([lane.position for lane in loaded_map.left_lane], dtype=np.float64)
        right_lane = np.asarray([lane.position for lane in loaded_map.right_lane], dtype=np.float64)
        path_left_point_indices = loaded_map.path_left_point_indices
        path_right_point_indices = loaded_map.path_right_point_indices

        points = []
        for i in range(0, len(path_left_point_indices)):
            p1 = loaded_map.left_lane[path_left_point_indices[i]].position
            p2 = loaded_map.right_lane[path_right_point_indices[i]].position
            points.append(0.5 * (p1 + p2))
        points.append(points[0])

        distances = [0]
        last = points[0]
        xs2 = [points[0][0]]
        ys2 = [points[0][1]]
        for point in points:
            dist = np.linalg.norm(point - last)
            if(dist > 2):
                last = point
                xs2.append(point[0])
                ys2.append(point[1])
                distances.append(dist + distances[-1])

        xs_spline = pacsim_pybind.CubicSpline(distances, xs2)
        ys_spline = pacsim_pybind.CubicSpline(distances, ys2)
        middle_line_length = distances[-1]

        left_positions = [lane.position for lane in loaded_map.left_lane]
        right_positions = [lane.position for lane in loaded_map.right_lane]
        left_poly = Polygon(left_positions)
        right_poly = Polygon(right_positions)
        if(left_poly.area > right_poly.area):
            outer_poly = left_poly
            inner_poly = right_poly
        else:
            outer_poly = right_poly
            inner_poly = left_poly
        prepare(outer_poly)
        prepare(inner_poly)

        return {
            "map": loaded_map,
            "start_orientation": np.array(start_orientation, dtype=np.float64),
            "left_lane": left_lane,
            "right_lane": right_lane,
            "path_left_point_indices": path_left_point_indices,
            "path_right_point_indices": path_right_point_indices,
            "xs_spline": xs_spline,
            "ys_spline": ys_spline,
            "middleLineLength": middle_line_length,
            "outer_poly": outer_poly,
            "inner_poly": inner_poly,
            "rangefinder": pacsim_pybind.Rangefinder(self.rangefinder_angles, left_lane, right_lane),
        }

    def _track_cache_key(self, mapFile, flipY):
        return (os.path.abspath(mapFile), bool(flipY))

    def _get_track_cache_entry(self, mapFile, flipY):
        key = self._track_cache_key(mapFile, flipY)
        entry = self.trackCache.get(key)
        if(entry is None):
            entry = self._build_track_cache_entry(mapFile, flipY)
            self.trackCache[key] = entry
        return entry

    def _vehicle_footprint_on_track(self, position, heading, outer_poly, inner_poly):
        """Return whether every vehicle corner is inside the drivable track area."""
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)
        corner_xs = (
            position[0]
            + cos_heading * self.carCornerOffsets[:, 0]
            - sin_heading * self.carCornerOffsets[:, 1]
        )
        corner_ys = (
            position[1]
            + sin_heading * self.carCornerOffsets[:, 0]
            + cos_heading * self.carCornerOffsets[:, 1]
        )
        return bool(np.all(contains_xy(outer_poly, corner_xs, corner_ys) & ~contains_xy(inner_poly, corner_xs, corner_ys)))

    def _preload_track_cache(self, mapFiles):
        for mapFile in mapFiles:
            self._get_track_cache_entry(mapFile, False)
            self._get_track_cache_entry(mapFile, True)

    @profile
    def _get_obs(self, ranges, velocity, rpm, steer, imu, cam):
        retDict = {
            "ranges": ranges,
            "velocity": velocity,
            "rpm": rpm,
            "steer": steer,
            "imu": imu,
        }
        if(self.outputLastActions):
            retDict["last_action"] = self.lastActionObservation.copy()
        if(self.outputRangefinder):
            retDict["ranges"] = ranges
        if(self.includeCameraObs and self.useCamSim):
            for camKey in self.cameraObsKeys:
                retDict[camKey] = self.emptyImageCode
            for i, image in enumerate(cam):
                if(i >= len(self.cameraObsKeys)):
                    break
                if(i < len(self.cameraEnabled) and (not self.cameraEnabled[i])):
                    continue
                retDict[self.cameraObsKeys[i]] = image.transpose(2,1,0)
        elif(self.includeCameraObs):
            for camKey in self.cameraObsKeys:
                retDict[camKey] = self.emptyImageCode
        return retDict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0.0,0.0,0.0])
        self.currentSteer = 0.0
        self.currentTorque = 0.0
        self.toruqes = [0.0,0.0,0.0,0.0]
        self.cameraImages = None
        self.lastActionObservation = np.zeros(5, dtype=np.float32)
        self.lastActionArray = np.zeros(5)
        self.prevActionArray = np.zeros(5)
        # self.arcLocalization = 0.0
        self.rays = []
        # print(self.np_random)
        # print(self.np_random)
        # print(self.np_random.integers(0, 99999))
        self.trackNr = self.np_random.integers(0, 999999)
        # self.trackNr = 78091
        # print("reset with trackNr {0}".format(self.trackNr))
        if(self.useComplexModel):
            self.model = pacsim_pybind.VehicleModel4Wheel()
        else:
            self.model = pacsim_pybind.VehicleModel()
        vehicle_model_config = self._resolve_existing_path(
            None,
            os.path.join(self.pacsimSourceRoot, "config", "vehicleModel.yaml"),
            "vehicle_model_config",
        )
        c = pacsim_pybind.Config(vehicle_model_config)
        c2 = c.getElement("vehicle_model")
        self.model.readConfig(c2)

        self.time = 0.0
        if(options is None):
            options = {}
        self.deadTime = float(options.get("dead_time", self.defaultDeadTime))
        self.steeringDeadTime = float(options.get(
            "steering_dead_time",
            options.get("dead_time_steering", options.get("dead_time", self.defaultSteeringDeadTime)),
        ))
        self.torqueDeadTime = float(options.get(
            "torque_dead_time",
            options.get("dead_time_torque", options.get("dead_time", self.defaultTorqueDeadTime)),
        ))

        self.deadTimeSteering = pacsim_pybind.ScalarDeadtime(self.steeringDeadTime)
        self.deadTimeRPMSetpoints = pacsim_pybind.WheelsDeadtime(self.torqueDeadTime)
        self.deadTimeMaxTorques = pacsim_pybind.WheelsDeadtime(self.torqueDeadTime)
        self.deadTimeMinTorques = pacsim_pybind.WheelsDeadtime(self.torqueDeadTime)

        self.start_orientation = np.array([0,0,0])

        mapFiles = self.defaultMapFiles
        try:
            mapFiles = options["map_files"]
        except:
            mapFiles = mapFiles

        numStartPoints = 10
        noAugment = bool(options.get("noAugment", False))
        selected_start = None
        last_start = None
        for startAttempt in range(1, self.maxStartPlacementAttempts + 1):
            mapFile = mapFiles[self.np_random.integers(0, len(mapFiles))]
            flipY = False if noAugment else self.np_random.random() < 0.5
            flipX = False if noAugment else self.np_random.random() < 0.5
            quartile = 0 if noAugment else self.np_random.integers(0, numStartPoints)
            trackEntry = self._get_track_cache_entry(mapFile, flipY)
            segmentPoint = trackEntry["middleLineLength"] * quartile / numStartPoints
            start_position = np.array([
                trackEntry["xs_spline"](segmentPoint),
                trackEntry["ys_spline"](segmentPoint),
                0.0,
            ])
            start_angle = np.arctan2(
                trackEntry["ys_spline"].derivative(segmentPoint),
                trackEntry["xs_spline"].derivative(segmentPoint),
            )
            base_orientation = trackEntry["start_orientation"].copy()
            heading = base_orientation[2] + start_angle + (np.pi if flipX else 0.0)
            last_start = (
                mapFile,
                flipY,
                flipX,
                quartile,
                trackEntry,
                segmentPoint,
                start_position,
                base_orientation,
                heading,
                startAttempt,
            )
            if(self._vehicle_footprint_on_track(
                start_position,
                heading,
                trackEntry["outer_poly"],
                trackEntry["inner_poly"],
            )):
                selected_start = last_start
                break

        if(selected_start is None):
            selected_start = last_start
            self.startPoseValid = False
            if(self.verbose):
                print(
                    "Warning: no valid start pose found after {0} attempts; using the last "
                    "candidate so the episode can terminate normally.".format(self.maxStartPlacementAttempts)
                )
        else:
            self.startPoseValid = True

        (
            mapFile,
            flipY,
            self.flipX,
            quartile,
            trackEntry,
            segmentPoint,
            start_position,
            self.start_orientation,
            start_heading,
            startAttempt,
        ) = selected_start
        self.start_orientation[2] = start_heading
        self.mapFile = mapFile
        self.trackName = os.path.splitext(os.path.basename(mapFile))[0]
        self.currentTrackNr = self.trackNr
        if(self.verbose):
            print(
                "TrackNr {0}, mapFile {1}, flipY {2}, flipX {3}, startSegment {4}, startAttempts {5}".format(
                    self.trackNr,
                    mapFile,
                    flipY,
                    self.flipX,
                    quartile,
                    startAttempt,
                )
            )
        self.map = trackEntry["map"]
        self.left_lane = trackEntry["left_lane"]
        self.right_lane = trackEntry["right_lane"]
        self.path_left_point_indices = trackEntry["path_left_point_indices"]
        self.path_right_point_indices = trackEntry["path_right_point_indices"]
        self.xs_spline = trackEntry["xs_spline"]
        self.ys_spline = trackEntry["ys_spline"]
        self.middleLineLength = trackEntry["middleLineLength"]
        self.outer_poly = trackEntry["outer_poly"]
        self.inner_poly = trackEntry["inner_poly"]
        self.rangefinder = trackEntry["rangefinder"]

        self.startArc = segmentPoint
        self.arcLocalization = segmentPoint
        self.lastArc = segmentPoint
        endArcDist = 7.0
        if(self.flipX):
            endArcDist *= -1.0
        self.endArc = (self.startArc+endArcDist) % self.middleLineLength
        # if(self.flipX):
        #     self.endArc = (self.startArc-10.0) % self.middleLineLength
        self.model.setOrientation(self.start_orientation)
        # print(self.model.getOrientation())
        self.model.setPosition(start_position)
        self.orientation = self.start_orientation
        self.position = self.model.getPosition()


        self.trackNr += 1


        if(self.useCamSim):
            self.cppCameraSim.setTrackAndCones(self.map)
            image = self.renderFrames(self.position, self.orientation, self.model.getSteeringWheelAngle(), self.model.getWheelOrientations())
            self.cameraImages = image

        self.rewards = []

        self.lastAfterLine = False
        self.odometer = 0
        self.lapCount = 0
        self.frameCounter = 0
        ranges, self.rays = self.rangefinder.rays(self.position, self.orientation)
        ranges = np.array(ranges)
        ranges = np.clip(ranges / self.maxRange,0.0,1.0)
        ret = self._get_obs(ranges, np.zeros(2), np.zeros(4), np.zeros(1), np.zeros(3), self.cameraImages)
        info  = {"laptime" : 0.0, "invalid_start_pose": not self.startPoseValid}

        return ret, info

    # @profile
    def renderFrames(self, position, orientation, steeringAngle, wheelOrientation):
        wheelOrientsArray = np.array([wheelOrientation.FL, wheelOrientation.FR, wheelOrientation.RL, wheelOrientation.RR], dtype=np.float64)

        t0 = None
        if(self.profileRender and self.useCamSim):
            t0 = time.perf_counter()

        images = self.cppCameraSim.render(position, orientation, steeringAngle, wheelOrientsArray)
        if(t0 is not None):
            dtMs = (time.perf_counter() - t0) * 1000.0
            self.profileRenderSamples += 1
            self.profileRenderMsSum += dtMs
            self.profileRenderMsMin = min(self.profileRenderMsMin, dtMs)
            self.profileRenderMsMax = max(self.profileRenderMsMax, dtMs)

            if(self.verbose and (self.profileRenderSamples % self.profileRenderEvery) == 0):
                avgMs = self.profileRenderMsSum / self.profileRenderSamples
                print(
                    "[pacsimEnv] cppCameraSim.render last={0:.2f} ms avg={1:.2f} ms min={2:.2f} ms max={3:.2f} ms samples={4}".format(
                        dtMs,
                        avgMs,
                        self.profileRenderMsMin,
                        self.profileRenderMsMax,
                        self.profileRenderSamples,
                    )
                )

        retImages = []
        for img in images:
            retImages.append(np.ascontiguousarray(img, dtype=np.uint8))

        return retImages
        # imageFront = self.panda3dRenderer.imbufclass.get_rgb_array()
        # imageFront = np.ascontiguousarray(imageFront, dtype=np.uint8)

        # return imageFront

    def rotMat2d(self, angle):
        ret = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return ret

    def render(self):
        import cv2
        import matplotlib as mpl
        mpl.use('Agg', force=True)
        mpl.rcParams['figure.dpi'] = 300
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # ax.plot([1, 2, 3], [4, 5, 6])

        _, self.rays = self.rangefinder.rays(self.position, self.orientation)

        left_pos = []
        for i in self.map.left_lane:
            left_pos.append(np.array(i.position[0:2]))
        left_pos.append(np.array(self.map.left_lane[0].position[0:2]))
        left_pos = np.array(left_pos)

        right_pos = []
        for i in self.map.right_lane:
            right_pos.append(np.array(i.position[0:2]))
        right_pos.append(np.array(self.map.right_lane[0].position[0:2]))
        right_pos = np.array(right_pos)


        ax.plot(left_pos.T[0], left_pos.T[1], "b", linewidth=1.5, markersize=1.5)
        ax.plot(right_pos.T[0], right_pos.T[1], "y", linewidth=1.5, markersize=1.5)
        ax.plot(left_pos.T[0], left_pos.T[1], "ob", linewidth=1.5, markersize=1.5)
        ax.plot(right_pos.T[0], right_pos.T[1], "oy", linewidth=1.5, markersize=1.5)

        for i in self.rays:
            ax.plot([i[0][0],i[0][0]+i[1][0]*i[2]], [i[0][1],i[0][1]+i[1][1]*i[2]],"r", linewidth=1)

        ego_pos = self.position
        ax.plot([ego_pos[0]], [ego_pos[1]],"ok",markersize=1)

        pointsCar = []
        r = self.rotMat2d(self.orientation[2])
        pointsCar.append(np.array(self.position[0:2] + r @ self.pFL[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pRL[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pRR[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pFR[0:2]))
        pointsCar.append(pointsCar[0])
        # print(pointsCar)
        pointsCar = np.array(pointsCar)
        # boxCar = []
        # print(pointsCar)
        # boxCar.append([[pointsCar[0][0], pointsCar[1][0]], [pointsCar[0][1], pointsCar[1][1]]])
        # boxCar.append([[pointsCar[1][0], pointsCar[2][0]], [pointsCar[1][1], pointsCar[2][1]]])
        # boxCar.append([[pointsCar[2][0], pointsCar[3][0]], [pointsCar[2][1], pointsCar[3][1]]])
        # boxCar.append([[pointsCar[3][0], pointsCar[0][0]], [pointsCar[3][1], pointsCar[0][1]]])
        # boxCar = np.array(boxCar)
        # print(boxCar)
        # print(boxCar.T)
        # # for p in pointsCar.T:
        # print(pointsCar.T[0])
        ax.plot(pointsCar.T[0], pointsCar.T[1], "-k", markersize=1.0, linewidth=0.7)
            

        midXs = []
        midYs = []
        for i in range(0, len(self.path_left_point_indices)):
            p1 = self.map.left_lane[self.path_left_point_indices[i]].position
            p2 = self.map.right_lane[self.path_right_point_indices[i]].position
            p3 = 0.5*(p1+p2)
            midXs.append(p3[0])
            midYs.append(p3[1])
            # print(p3)
        # ax.plot(midXs, midYs, "x")
        # for i in range(0,len(midXs)):
        #     # print(i)
        #     ax.annotate(i, (midXs[i], midYs[i]),fontsize=5)

        x_splines = np.linspace(0, self.middleLineLength, 300)
        splineXs = []
        splineYs = []
        for i in x_splines:
            splineXs.append(self.xs_spline(i))
            splineYs.append(self.ys_spline(i))
            # print(self.xs_spline(i))
        # ax.plot(splineXs, splineYs)

        # ax.plot([self.xs_spline(self.arcLocalization)], [self.ys_spline(self.arcLocalization)], "x")
        # print("end arc" + str(self.endArc))
        # ax.plot([self.xs_spline(self.startArc)], [self.ys_spline(self.startArc)], "o")
        # ax.plot([self.xs_spline((self.endArc-0.0)%self.middleLineLength)], [self.ys_spline((self.endArc-0.0)%self.middleLineLength)], "or")
        # ax.plot([self.xs_spline((self.endArc+15.0)%self.middleLineLength)], [self.ys_spline((self.endArc+15.0)%self.middleLineLength)], "or")
        # print(self.xs_spline(self.endArc))
        # print(self.xs_spline(self.startArc))
        # print(self.xs_spline((self.endArc+0.0)%self.middleLineLength))
        # print(self.ys_spline((self.endArc+0.0)%self.middleLineLength))

        ax.set_aspect('equal')

        fig.canvas.draw()

        # fig.savefig('full_figure.png')
        # fig.canvas.print_png(filename)
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        fig.clear()
        plt.close(fig)
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        timestring = 't=' + str(round(self.time,3))
        # print(timestring)
        # print(image)
        # print(image.shape)
        image = np.array(image, copy=True)
        image = cv2.putText(image, timestring, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        vel = self.model.getVelocity()[0:2]
        velocityString = "v=" + str([round(vel[0],2), round(vel[1],2)])
        image = cv2.putText(image, velocityString, (50,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        yawrate = self.model.getAngularVelocity()[2]
        yawrateString = "yawrate=" + str(round(yawrate,3))
        image = cv2.putText(image, yawrateString, (50,150), font, 
                fontScale, color, thickness, cv2.LINE_AA)

        
        trackNrString = "trackNumber=" + str(self.trackNr)
        image = cv2.putText(image, trackNrString, (500,50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        acc = self.model.getAcceleration()[0:2]
        accString = "acc=" + str([round(acc[0],2), round(acc[1],2)])
        image = cv2.putText(image, accString, (500,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        steeringString = "steering=" + str(round(self.currentSteer,3))
        image = cv2.putText(image, steeringString, (1000,50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        # torqueString = "torque=" + str(round(self.currentTorque,2))
        flString = "torque FL=" + str(round(self.toruqes[0],2))
        frString = "torque FR=" + str(round(self.toruqes[1],2))
        image = cv2.putText(image, flString, (1000,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, frString, (1320,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        rlString = "torque RL=" + str(round(self.toruqes[2],2))
        rrString = "torque RR=" + str(round(self.toruqes[3],2))
        image = cv2.putText(image, rlString, (1000,150), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, rrString, (1320,150), font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        # return
    
    # @profile
    def getReward(self, finished, collided):


        episode_rew = 0
        curVel = self.model.getVelocity()
        omega = self.model.getAngularVelocity()

        def cross(a,b):
            ret = np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
            return ret
        
        vFL = curVel + cross(omega, self.pFL)
        vFR = curVel + cross(omega, self.pFR)
        vRL = curVel + cross(omega, self.pRL)
        vRR = curVel + cross(omega, self.pRR)

        alphaFL = 0.0
        alphaFR = 0.0
        alphaRL = 0.0
        alphaRR = 0.0
        if(np.abs(curVel[0] > 2)):
            if(np.abs(vFL[0] > 0.1)):
                alphaFL = np.abs(np.arctan2(vFL[1], vFL[0]) - self.currentSteer)
            if(np.abs(vFR[0] > 0.1)):
                alphaFR = np.abs(np.arctan2(vFR[1], vFR[0]) - self.currentSteer)
            if(np.abs(vRL[0] > 0.1)):
                alphaRL = np.abs(np.arctan2(vRL[1], vRL[0]))
            if(np.abs(vRR[0] > 0.1)):
                alphaRR = np.abs(np.arctan2(vRR[1], vRR[0]))
        # print("Alpha FL: {0}, FR: {1}, RL: {2}, RR: {3}".format(alphaFL, alphaFR, alphaRL, alphaRR))


        kappaFL = 0.0
        kappaFR = 0.0
        kappaRL = 0.0
        kappaRR = 0.0
        gearRatio = 12.23
        wheelRadius = 0.206
        rpm2ms = wheelRadius * 2.0 * np.pi / (gearRatio * 60.0)
        def getSlip(wheelspeed, vx):
            eps = 0.0001
            ret = np.abs(wheelspeed-vx) / max(np.abs(vx),eps)
            return ret

        if(np.abs(curVel[0] > 2)):
            kappaFL = getSlip(self.wheelspeeds.FL*rpm2ms, vFL[0])
            kappaFR = getSlip(self.wheelspeeds.FR*rpm2ms, vFR[0])
            kappaRL = getSlip(self.wheelspeeds.RL*rpm2ms, vRL[0])
            kappaRR = getSlip(self.wheelspeeds.RR*rpm2ms, vRR[0])

        # print("Long slip FL: {0}, FR: {1}, RL: {2}, RR: {3}".format(kappaFL, kappaFR, kappaRL, kappaRR))

        kappa = np.max([np.abs(kappaFL), np.abs(kappaFR), np.abs(kappaRL), np.abs(kappaRR)])
        if(not self.useComplexModel):
            kappa = 0.0
        # kappa = 0.0

        alpha = max(min(alphaFL, alphaFR), min(alphaRL, alphaRR))
        alpha = np.rad2deg(alpha)

        def signed_modulo_distance(a, b, m):
            return ((b - a + m/2) % m) - m/2

        sdot = signed_modulo_distance(self.lastArc, self.arcLocalization, self.middleLineLength) / self.interval

        if(self.flipX):
            sdot = -1.0 * sdot

        actionDelta = self.lastActionArray - self.prevActionArray
        actionRateReg = (5*np.abs(actionDelta[0]/self.maxSteering) + np.abs(actionDelta[1]/self.maxTorque) + np.abs(actionDelta[2]/self.maxTorque) + np.abs(actionDelta[3]/self.maxTorque) + np.abs(actionDelta[4]/self.maxTorque)) / self.interval
        
        r_progress = sdot
        r_tracking = -np.abs(self.curvCoords[1])
        r_finish = 1.0 if finished else 0.0
        r_collition = -1.0 if collided else 0.0
        r_stand = -1.0 if (curVel[0] < 2.0) else 0.0
        r_slipAngle = -np.abs(alpha)
        r_slipRatio = -kappa
        r_actionRate = -actionRateReg
        r_lateral_consistency = -max(0,-((self.lastActionArray[1] - self.lastActionArray[2]) * (self.lastActionArray[3] - self.lastActionArray[4])))
        r_longitudinal_consistency = -max(0,-((self.lastActionArray[1] + self.lastActionArray[2]) * (self.lastActionArray[3] + self.lastActionArray[4])))

        total_reward = self.lambda_progress * r_progress + self.lambda_tracking * r_tracking + self.lambda_finish * r_finish + self.lambda_collition * r_collition + self.lambda_stand * r_stand + self.lambda_slipAngle * r_slipAngle + self.lambda_slipRatio * r_slipRatio + self.lambda_actionRate * r_actionRate + self.lambda_lateral_consistency * r_lateral_consistency + self.lambda_longitudinal_consistency * r_longitudinal_consistency

        return total_reward

    @profile
    def step(self, actions):
        profile_step = self.profileEnvStep
        step_t0 = time.perf_counter() if profile_step else None
        section_t0 = step_t0

        def mark_section(name):
            nonlocal section_t0
            if(not profile_step):
                return
            now = time.perf_counter()
            self._profile_env_step_add(name, now - section_t0)
            section_t0 = now

        dt = 1.0/1000.0
        actions = np.asarray(actions, dtype=np.float32)

        wmaxTorque = pacsim_pybind.Wheels()
        wminTorque = pacsim_pybind.Wheels()

        curVel = np.linalg.norm(self.model.getVelocity())
        wmaxRPM = pacsim_pybind.Wheels()

        self.toruqes = np.array([self.maxTorque*actions[1], self.maxTorque*actions[2], self.maxTorque*actions[3], self.maxTorque*actions[4]])
        if(self.useComplexModel):
            if(self.toruqes[0] >= 0):
                wmaxRPM.FL = 20000.0
                wmaxTorque.FL = self.toruqes[0]
                wminTorque.FL = 0.0
            else:
                wmaxRPM.FL = 0.0
                wmaxTorque.FL = 0.1
                wminTorque.FL = self.toruqes[0]

            if(self.toruqes[1] >= 0):
                wmaxRPM.FR = 20000.0
                wmaxTorque.FR = self.toruqes[1]
                wminTorque.FR = 0.0
            else:
                wmaxRPM.FR = 0.0
                wmaxTorque.FR = 0.1
                wminTorque.FR = self.toruqes[1]

            if(self.toruqes[2] >= 0):
                wmaxRPM.RL = 20000.0
                wmaxTorque.RL = self.toruqes[2]
                wminTorque.RL = 0.0
            else:
                wmaxRPM.RL = 0.0
                wmaxTorque.RL = 0.1
                wminTorque.RL = self.toruqes[2]

            if(self.toruqes[3] >= 0):
                wmaxRPM.RR = 20000.0
                wmaxTorque.RR = self.toruqes[3]
                wminTorque.RR = 0.0
            else:
                wmaxRPM.RR = 0.0
                wmaxTorque.RR = 0.1
                wminTorque.RR = self.toruqes[3]
        else:
            wmaxTorque.FL = self.toruqes[0]
            wmaxTorque.FR = self.toruqes[1]
            wmaxTorque.RR = self.toruqes[2]
            wmaxTorque.RR = self.toruqes[3]

        steering_action = self.maxSteering*actions[0]
        self.currentSteer = steering_action
        self.currentTorque = self.maxTorque*actions[1]

        self.deadTimeSteering.addVal(steering_action, self.time)
        self.deadTimeRPMSetpoints.addVal(wmaxRPM, self.time)
        self.deadTimeMaxTorques.addVal(wmaxTorque, self.time)
        self.deadTimeMinTorques.addVal(wminTorque, self.time)
        
        futureTime = self.interval * (self.frameCounter+1.0)
        wFric = pacsim_pybind.Wheels()
        wFric.FL = 1.0
        wFric.FR = 1.0
        wFric.RL = 1.0
        wFric.RR = 1.0
        mark_section("action_setup")

        # batched in c++ to avoid repeated overhead from python<->c++ context switches
        self.time = self.model.forwardIntegrateControlFrame(
            self.time,
            futureTime,
            dt,
            self.deadTimeSteering,
            self.deadTimeRPMSetpoints,
            self.deadTimeMaxTorques,
            self.deadTimeMinTorques,
            wFric,
        )
        mark_section("integrate_loop")
        position = self.model.getPosition()
        orientation = self.model.getOrientation()
        currentSteer = self.model.getSteeringWheelAngle()
        self.position = position
        self.orientation = orientation
        pose = np.array([position[0], position[1], orientation[2]])
        curVel = np.linalg.norm(self.model.getVelocity())
        self.odometer += curVel * self.interval
        mark_section("pose_velocity_getters")

        ranges = self.rangefinder.normalizedDistances(self.position, self.orientation, self.maxRange)
        mark_section("rangefinder")

        if(self.useCamSim):
            images = self.renderFrames(position, orientation, self.model.getSteeringWheelAngle(), self.model.getWheelOrientations())
            self.cameraImages = images
        self.frameCounter += 1
        mark_section("render_if_enabled")

        vel = self.model.getVelocity()
        rot = self.model.getAngularVelocity()
        acceleration = self.model.getAcceleration()
        self.acceleration = acceleration[0:2]
        velArray = np.array([vel[0],vel[1],rot[2]])
        rpms = self.model.getWheelspeeds()
        curTorques = self.model.getTorques()
        self.wheelspeeds = rpms
        rpmArray = np.array([rpms.FL, rpms.FR, rpms.RL, rpms.RR])
        self.prevActionArray = self.lastActionArray
        self.lastActionArray = np.array([steering_action, self.toruqes[0], self.toruqes[1], self.toruqes[2], self.toruqes[3]])
        self.lastActionObservation = np.clip(actions, -1.0, 1.0).astype(np.float32, copy=True)

        velArrayNorm = np.array([vel[0]/self.maxSpeed,vel[1]/self.maxSpeed])
        rpmArrayOut = rpmArray / self.maxRpm
        steerNormArray  = np.array([np.clip(currentSteer/self.maxSteering,-1.0,1.0)])

        imuDataNorm = np.array([acceleration[0]/self.maxImuAcceleration, acceleration[1]/self.maxImuAcceleration, rot[2]/self.maxYawRate])
        imuNormArray = np.array(np.clip(imuDataNorm,-1.0,1.0))

        ret = self._get_obs(ranges, velArrayNorm, rpmArrayOut, steerNormArray, imuNormArray, self.cameraImages)
        info  = {"laptime" : 0.0}
        mark_section("sensor_getters_obs")

        self.curvCoords = pacsim_pybind.findCurvlinearCoords(self.xs_spline, self.ys_spline, self.middleLineLength, self.position[0], self.position[1])
        self.arcLocalization = self.curvCoords[0]
        mark_section("curvilinear_coords")
        
        collided = not self._vehicle_footprint_on_track(
            self.position,
            self.orientation[2],
            self.outer_poly,
            self.inner_poly,
        )
        mark_section("collision_check")

        def is_modulo_greater(a, b, mod):
            """
            Returns True if b comes after a in modulo `mod` space.
            """
            return (b - a) % mod < mod // 2


        afterLine = is_modulo_greater(self.endArc, self.arcLocalization, self.middleLineLength)
        if(self.flipX):
            afterLine = not afterLine
        crossedLine = (self.odometer > 20.0) and afterLine and (not self.lastAfterLine)
        if(afterLine and (self.odometer <= 20.0) and (not self.lastAfterLine)):
            self.timingStart = self.time
        reward = self.getReward(crossedLine, collided)
        self.lastAfterLine = afterLine
        self.lastArc = self.arcLocalization

        timeout = self.time > 120.0 or ((self.odometer < 1.0) and (self.time > 5.0)) or ((self.odometer < 3.0) and (self.time > 10.0)) or ((self.odometer < 10.0) and (self.time > 30.0)) or ((self.odometer < 20.0) and (self.time > 60.0))
        terminated = collided or crossedLine or timeout
        info["collided"] = bool(collided)
        info["crossed_line"] = bool(crossedLine)
        info["timeout"] = bool(timeout)
        if(self.printStepStatus and self.verbose and ((self.frameCounter % 100) == 3)):
            actionDenormalized = np.array([self.maxSteering*actions[0], self.maxTorque*actions[1], self.maxTorque*actions[2], self.maxTorque*actions[3], self.maxTorque*actions[4]])
            # print("Time {0}, action: {1}, pose: {2}, velocity: {3}".format(self.time, actionDenormalized, pose, velArray))
            print("Time {0:.3f}, action: {1}, pose: {2}, velocity: {3}".format(
                self.time,
                np.array2string(actionDenormalized, precision=2, suppress_small=True),
                np.array2string(pose, precision=2, suppress_small=True),
                np.array2string(velArray, precision=2, suppress_small=True)
            ))
            # print("Cur torques FL {0}, FR: {1}, RL: {2}, RR: {3}".format(curTorques.FL, curTorques.FR, curTorques.RL, curTorques.RR))
        if(terminated and self.verbose):
            def b2s(arg, col="red"):
                if(arg):
                    return termcolor.colored("True", col)
                else:
                    return "False"
            print("Time {0:.3f}, TrackNr {1}, Track {2}, Collided {3}, CrossedLine {4}, Timeout {5}, Pose {6}, Velocity {7}".format(self.time, self.currentTrackNr, self.trackName, b2s(collided), b2s(crossedLine, col="green"), b2s(timeout), np.array2string(pose, precision=2, suppress_small=True), np.array2string(velArray, precision=3, suppress_small=True)))
        if(crossedLine):
            info["laptime"] = self.time - self.timingStart
        if(timeout and self.verbose):
            print("Timeout")
        truncated = False
        mark_section("reward_done_info")
        if(profile_step):
            self._profile_env_step_add("step_total", time.perf_counter() - step_t0)
        return ret, reward, terminated, truncated, info


def _build_pacsim_env_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run a pacsimEnv Vulkan renderer smoke test.")
    parser.add_argument("--camera-renderer", choices=("vulkan",), default="vulkan")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--action",
        type=float,
        nargs=5,
        default=[0.0, 0.8, 0.8, 0.8, 0.8],
        metavar=("STEER", "FL", "FR", "RL", "RR"),
    )
    parser.add_argument("--camera-config-file", default=None)
    parser.add_argument("--map-file", default="/root/workspace/tracks/FSE22.yaml")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--profile-render", action="store_true")
    parser.add_argument("--include-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steering-dead-time", type=float, default=0.0)
    parser.add_argument("--torque-dead-time", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _run_pacsim_env_cli():
    args = _build_pacsim_env_arg_parser().parse_args()
    env_args = {
        "cam_sim": True,
        "camera_renderer": args.camera_renderer,
        "camera_config_file": args.camera_config_file,
        "use_shadows": not args.no_shadows,
        "profile_render": args.profile_render,
        "include_last_action": args.include_last_action,
        "steering_dead_time": args.steering_dead_time,
        "torque_dead_time": args.torque_dead_time,
        "print_step_status": not args.quiet,
    }

    env = pacsimEnv(env_args)
    reset_options = {
        "noAugment": True,
        "map_files": [args.map_file],
    }
    env.reset(seed=args.seed, options=reset_options)
    action = np.array(args.action, dtype=env.action_space.dtype)

    max_speed = 0.0
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(action)
        max_speed = max(max_speed, float(np.linalg.norm(env.model.getVelocity())))
        if(terminated or truncated):
            env.reset(seed=args.seed, options=reset_options)

    print(
        "pacsimEnv renderer={0} steps={1} max_speed={2:.3f} m/s".format(
            args.camera_renderer, args.steps, max_speed
        )
    )


if __name__ == "__main__":
    _run_pacsim_env_cli()
