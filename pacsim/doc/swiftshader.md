# SwiftShader Runtime

pacsim should not depend on a SwiftShader copy extracted from another product.
For reproducible builds, create a small SwiftShader runtime artifact from the
canonical upstream source at a pinned revision, then pass that artifact into the
ROS or Python packaging step explicitly.

The helper script builds the upstream `vk_swiftshader` target and stages:

```text
vk_swiftshader_icd.json
libvk_swiftshader.so | libvk_swiftshader.dylib | vk_swiftshader.dll
LICENSE.txt
AUTHORS.txt
CONTRIBUTORS.txt
swiftshader_build_manifest.json
```

The staged ICD JSON is rewritten so `library_path` points to the sibling runtime
library. That keeps the artifact relocatable.

## Build The Runtime

Linux and macOS:

```bash
python3 packaging/swiftshader/build_swiftshader.py \
  --prefix /tmp/pacsim-swiftshader
```

Windows from a Visual Studio developer shell:

```powershell
py packaging\swiftshader\build_swiftshader.py `
  --prefix C:\tmp\pacsim-swiftshader `
  --generator "Visual Studio 17 2022"
```

The default source is:

```text
https://swiftshader.googlesource.com/SwiftShader
```

The default pinned revision is recorded in
`packaging/swiftshader/build_swiftshader.py`.
Changing it should be intentional and should regenerate the runtime artifact for
every platform.

By default the SwiftShader source and build tree are kept under
`~/.cache/pacsim/swiftshader`. CI can pass explicit `--source-dir` and
`--build-dir` paths if it wants isolated workspaces.

## Use With ROS

The ROS package can stay lightweight and consume the staged runtime only on
machines that need software rendering:

```bash
source install/setup.bash
export PACSIM_FORCE_SWIFTSHADER=1
export PACSIM_SWIFTSHADER_ICD=/tmp/pacsim-swiftshader/vk_swiftshader_icd.json
ros2 launch pacsim example.launch.py
```

You can also place the staged files next to the installed pacsim library:

```text
install/pacsim/lib/pacsim/swiftshader/vk_swiftshader_icd.json
install/pacsim/lib/pacsim/swiftshader/libvk_swiftshader.so
```

Then `PACSIM_SWIFTSHADER_ICD` is not required; `PACSIM_FORCE_SWIFTSHADER=1` is
enough to force software rendering.

## Use With Python Wheels

Build the SwiftShader runtime first, then explicitly bundle it into the wheel:

```bash
python3 packaging/swiftshader/build_swiftshader.py \
  --prefix /tmp/pacsim-swiftshader

python3 -m pip wheel . -w dist --no-deps \
  --config-settings=cmake.define.PACSIM_BUILD_ROS=OFF \
  --config-settings=cmake.define.PACSIM_BUILD_PYTHON=ON \
  --config-settings=cmake.define.PACSIM_FORCE_SWIFTSHADER=ON \
  --config-settings=cmake.define.PACSIM_BUNDLE_SWIFTSHADER=ON \
  --config-settings=cmake.define.PACSIM_SWIFTSHADER_ICD_JSON=/tmp/pacsim-swiftshader/vk_swiftshader_icd.json
```

For Linux release wheels, run `auditwheel repair` after building the wheel.

## Vulkan Loader

SwiftShader is a Vulkan ICD. pacsim still needs a Vulkan loader at runtime.
Linux ROS deployments can usually install the loader through the system package
manager. Python wheels may bundle the loader during wheel repair.

For fully self-contained Windows software-rendering packages, build with:

```powershell
py packaging\swiftshader\build_swiftshader.py `
  --prefix C:\tmp\pacsim-swiftshader `
  --generator "Visual Studio 17 2022" `
  --direct-loader-alias
```

That additionally stages `vulkan-1.dll` as a direct SwiftShader loader alias.
Use this only for software-only packages.
