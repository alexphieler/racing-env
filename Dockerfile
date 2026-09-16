# ROS2 base image
FROM rwthika/ros2-cuda:kilted-desktop-full-v25.08 AS base

# The base image sources this script from /root/.bashrc for every interactive
# shell. It prints a banner and runs several package, Python, and GPU probes,
# so remove that source line to keep shell startup quiet and fast.
RUN sed -i '\|^source /.version_information\.sh$|d' /root/.bashrc

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Install dependencies with apt
RUN apt update && \
  DEBIAN_FRONTEND=noninteractive apt install -y keyboard-configuration && \
  apt install -y git \
  apt-utils \
  software-properties-common \
  desktop-file-utils \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-pip \
  libpcap-dev \
  gnuplot \
  libboost-all-dev \
  libpcl-dev \
  libncurses5-dev libncursesw5-dev \
  ros-$ROS_DISTRO-yaml-cpp-vendor ros-$ROS_DISTRO-xacro ros-$ROS_DISTRO-foxglove-bridge \
  ros-$ROS_DISTRO-pcl-ros ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-diagnostic-updater \
  ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-transport-plugins \
  pybind11-dev \
  ffmpeg \
  libassimp-dev \
  build-essential \
  cmake \
  ninja-build \
  libeigen3-dev \
  libshaderc-dev \
  libvulkan-dev \
  libvulkan1 \
  vulkan-tools \
  strace \
  ripgrep \
  sd \
  jq \
  yq

RUN pip install pyyaml \
  subprocess32 \
  "numpy<2" \
  scipy \
  matplotlib \
  numba \
  tqdm \
  rosnumpy \
  ruamel.yaml \
  panda3d \
  panda3d-gltf \
  panda3d-simplepbr \
  pandas

RUN apt update
RUN apt install -y \
    gdb \
    gdbserver \
    ros-$ROS_DISTRO-backward-ros \
    vim \
    tmux \
    htop \
    bash-completion \
    graphviz

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc
RUN echo "source /root/workspace/pacsim_ws/install/setup.bash" >> /root/.bashrc
RUN echo "export COLCON_DEFAULTS_FILE=/root/workspace/configs/colcon_config.yaml" >> /root/.bashrc

ARG PACSIM_BUILD_ROS=ON
ARG PACSIM_BUILD_PYTHON=ON

ENV SHELL /bin/bash
ENV LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib:/opt/ros/${ROS_DISTRO}/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV PACSIM_SWIFTSHADER_ICD=/opt/pacsim-swiftshader/vk_swiftshader_icd.json
SHELL ["/bin/bash", "-c"]

RUN printf "/opt/ros/%s/lib\n/opt/ros/%s/lib/x86_64-linux-gnu\n" "$ROS_DISTRO" "$ROS_DISTRO" > /etc/ld.so.conf.d/ros2.conf && ldconfig

RUN pip install --ignore-installed gymnasium tensorboard tqdm termcolor tyro line_profiler rich
RUN pip install --ignore-installed torch torchvision torchmetrics
RUN pip install --ignore-installed matplotlib==3.7.0
RUN pip install onnx
RUN pip install einops moviepy==1.0.3 imageio requests "pillow<12"
# Keep NumPy and Shapely pip-owned.  The Debian Shapely binary is built against
# the distribution NumPy ABI, while later pip installs may otherwise select
# NumPy 2.x.  Reinstalling this pair at the end leaves one consistent ABI.
RUN pip install --upgrade --force-reinstall "numpy<2" "shapely>=2"

COPY pacsim/packaging/swiftshader/build_swiftshader.py /tmp/build_swiftshader.py
RUN python3 /tmp/build_swiftshader.py \
    --prefix /opt/pacsim-swiftshader \
    --source-dir /tmp/pacsim-swiftshader-src \
    --build-dir /tmp/pacsim-swiftshader-build \
    --generator Ninja \
    --absolute-library-path \
    --jobs 4 && \
  rm -rf /tmp/pacsim-swiftshader-src /tmp/pacsim-swiftshader-build /tmp/build_swiftshader.py

COPY pacsim /root/workspace/pacsim_ws/src/pacsim
WORKDIR /root/workspace/pacsim_ws
RUN source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DPACSIM_BUILD_ROS=${PACSIM_BUILD_ROS} \
    -DPACSIM_BUILD_PYTHON=${PACSIM_BUILD_PYTHON} \
    -DPACSIM_BUNDLE_SWIFTSHADER=ON \
    -DPACSIM_SWIFTSHADER_ICD_JSON=/opt/pacsim-swiftshader/vk_swiftshader_icd.json

ENV LD_LIBRARY_PATH=/root/workspace/pacsim_ws/install/pacsim/lib/pacsim:${LD_LIBRARY_PATH}

RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
  source /root/workspace/pacsim_ws/install/setup.bash && \
  PACSIM_FORCE_SWIFTSHADER=1 python3 -c "import gc, pacsim_pybind; sim = pacsim_pybind.VulkanCameraSim(16, 16); print('SwiftShader smoke test ok:', sim.width(), sim.height()); del sim; gc.collect()"

CMD ["/bin/bash"]
