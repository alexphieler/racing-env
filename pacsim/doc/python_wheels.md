# Python Wheels

pacsim can be built as a platform wheel with the Python extension and a bundled
SwiftShader ICD. The wheel layout is:

```text
pacsim/
  __init__.py
  pacsim_pybind.<abi>.so
  swiftshader/
    vk_swiftshader_icd.json
    libvk_swiftshader.so
pacsim_pybind.py
```

`pacsim_pybind.py` preserves the historical top-level `import pacsim_pybind`
entry point. New code can use `import pacsim`.

The wheel-only Python package files live under `packaging/python/` so the core
C++ and ROS source tree stays separate from Python packaging boilerplate.

Build a local wheel with a prepared SwiftShader runtime. See
[`doc/swiftshader.md`](swiftshader.md) for the reproducible cross-platform
runtime build.

```bash
python -m pip wheel . -w dist --no-deps \
  --config-settings=cmake.define.PACSIM_BUILD_ROS=OFF \
  --config-settings=cmake.define.PACSIM_BUILD_PYTHON=ON \
  --config-settings=cmake.define.PACSIM_BUNDLE_SWIFTSHADER=ON \
  --config-settings=cmake.define.PACSIM_SWIFTSHADER_ICD_JSON=/path/to/swiftshader/vk_swiftshader_icd.json
```

For a forced-software wheel, add:

```bash
--config-settings=cmake.define.PACSIM_FORCE_SWIFTSHADER=ON
```

Linux wheels built on a local distro need repair before publishing:

```bash
python -m auditwheel repair dist/pacsim-*.whl -w wheelhouse
```

For broad PyPI compatibility, build Linux wheels inside a manylinux image or
with cibuildwheel, then run/let cibuildwheel run auditwheel. A wheel built on a
new host can only be tagged for that host's glibc floor.

The bundled SwiftShader files should be produced from pinned SwiftShader source
in release CI and shipped with SwiftShader's license and notices.

## Local Build And Smoke Test

The commands below build the current Linux wheel from an explicitly prepared
SwiftShader runtime directory. The directory must contain
`vk_swiftshader_icd.json` and the matching runtime library.

```bash
cd /root/workspace/pacsim

python3 -m pip install --upgrade pip
python3 -m pip install build scikit-build-core pybind11 auditwheel
sudo apt-get update
sudo apt-get install -y patchelf

export PACSIM_SWIFTSHADER_ICD_JSON=/path/to/swiftshader/vk_swiftshader_icd.json

rm -rf dist wheelhouse
python3 -m pip wheel . -w dist --no-deps \
  --config-settings=cmake.define.PACSIM_BUILD_ROS=OFF \
  --config-settings=cmake.define.PACSIM_BUILD_PYTHON=ON \
  --config-settings=cmake.define.PACSIM_FORCE_SWIFTSHADER=ON \
  --config-settings=cmake.define.PACSIM_BUNDLE_SWIFTSHADER=ON \
  --config-settings=cmake.define.PACSIM_SWIFTSHADER_ICD_JSON="${PACSIM_SWIFTSHADER_ICD_JSON}"

python3 -m auditwheel repair dist/pacsim-*.whl -w wheelhouse
```

Install and test the repaired wheel:

```bash
python3 -m pip install --force-reinstall wheelhouse/pacsim-*.whl

python3 - <<'PY'
import pacsim
import pacsim_pybind

print("pacsim:", pacsim.__file__)
print("SwiftShader ICD:", pacsim.bundled_swiftshader_icd_path())
print("compat import:", hasattr(pacsim_pybind, "VulkanCameraSim"))

sim = pacsim.VulkanCameraSim(16, 16)
print("camera sim:", sim.width(), sim.height())
PY
```

For a local throwaway install without touching the active Python environment:

```bash
rm -rf /tmp/pacsim-wheel-target
python3 -m pip install --target /tmp/pacsim-wheel-target --no-deps wheelhouse/pacsim-*.whl

PYTHONPATH=/tmp/pacsim-wheel-target python3 - <<'PY'
import pacsim
sim = pacsim.VulkanCameraSim(16, 16)
print(pacsim.bundled_swiftshader_icd_path())
print(sim.width(), sim.height())
PY
```

## CI Build Notes

For PyPI, build inside manylinux through cibuildwheel instead of repairing a
wheel produced directly on a modern Ubuntu host:

```bash
python3 -m pip install cibuildwheel

export CIBW_CONFIG_SETTINGS="cmake.define.PACSIM_BUILD_ROS=OFF cmake.define.PACSIM_BUILD_PYTHON=ON cmake.define.PACSIM_FORCE_SWIFTSHADER=ON cmake.define.PACSIM_BUNDLE_SWIFTSHADER=ON cmake.define.PACSIM_SWIFTSHADER_ICD_JSON=/project/path/to/swiftshader/vk_swiftshader_icd.json"
python3 -m cibuildwheel --output-dir wheelhouse
```

The SwiftShader runtime path must exist inside the build container. In practice,
CI should build SwiftShader first or download a prepared SwiftShader artifact
for the platform, then pass that ICD path through
`PACSIM_SWIFTSHADER_ICD_JSON`.

## ROS Builds

ROS builds do not need SwiftShader at build time. Build the normal ROS package
without `PACSIM_SWIFTSHADER_ICD_JSON`:

```bash
source /opt/ros/kilted/setup.bash
colcon build --packages-select pacsim --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPACSIM_BUILD_ROS=ON \
  -DPACSIM_BUILD_PYTHON=OFF \
  -DPACSIM_FORCE_SWIFTSHADER=OFF
```

That binary uses hardware Vulkan first. If you want to force SwiftShader for a
ROS run without rebuilding pacsim, set both runtime variables:

```bash
source install/setup.bash
export PACSIM_FORCE_SWIFTSHADER=1
export PACSIM_SWIFTSHADER_ICD=/path/to/swiftshader/vk_swiftshader_icd.json
ros2 launch pacsim example.launch.py
```

Unset `PACSIM_FORCE_SWIFTSHADER` to return to hardware-first behavior.
