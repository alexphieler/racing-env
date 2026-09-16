cmake_minimum_required(VERSION 3.15)
project(pacsim LANGUAGES C CXX)

option(PACSIM_BUILD_ROS "Build ROS2 node, generated interfaces, and ament package" ON)
option(PACSIM_BUILD_PYTHON "Build Python bindings and install Python package files." OFF)
set(PACSIM_SWIFTSHADER_ICD_JSON "" CACHE FILEPATH
  "Optional path to vk_swiftshader_icd.json. Used as the software Vulkan fallback ICD.")
option(PACSIM_BUNDLE_SWIFTSHADER
  "Copy the SwiftShader ICD JSON and sibling runtime library next to pacsim targets when PACSIM_SWIFTSHADER_ICD_JSON is set."
  OFF)
option(PACSIM_FORCE_SWIFTSHADER
  "Skip hardware Vulkan selection and use the SwiftShader software fallback path."
  OFF)
set(PACSIM_PYTHON_PACKAGE "pacsim" CACHE STRING
  "Python package directory used by non-ROS wheel installs.")
set(PACSIM_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/packaging/python")
set(PACSIM_PYTHON_BINDINGS_SOURCE "${PROJECT_SOURCE_DIR}/src/bindings.cpp")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(assimp REQUIRED)
find_package(Vulkan REQUIRED)
find_package(tinyxml2 CONFIG QUIET)
find_package(shaderc CONFIG QUIET)
find_package(unofficial-shaderc CONFIG QUIET)

if(PACSIM_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)
endif()

set(PACSIM_YAML_CPP_LINK ${YAML_CPP_LIBRARIES})
if(TARGET yaml-cpp::yaml-cpp)
  set(PACSIM_YAML_CPP_LINK yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(PACSIM_YAML_CPP_LINK yaml-cpp)
endif()

set(PACSIM_ASSIMP_LINK ${ASSIMP_LIBRARIES})
if(TARGET assimp::assimp)
  set(PACSIM_ASSIMP_LINK assimp::assimp)
elseif(TARGET assimp)
  set(PACSIM_ASSIMP_LINK assimp)
endif()

set(PACSIM_SHADERC_LINK shaderc)
if(TARGET unofficial::shaderc::shaderc)
  set(PACSIM_SHADERC_LINK unofficial::shaderc::shaderc)
elseif(TARGET shaderc::shaderc)
  set(PACSIM_SHADERC_LINK shaderc::shaderc)
elseif(TARGET shaderc_shared)
  set(PACSIM_SHADERC_LINK shaderc_shared)
elseif(TARGET shaderc_combined)
  set(PACSIM_SHADERC_LINK shaderc_combined)
elseif(TARGET shaderc)
  set(PACSIM_SHADERC_LINK shaderc)
endif()

set(PACSIM_TINYXML2_LINK tinyxml2)
if(TARGET tinyxml2::tinyxml2)
  set(PACSIM_TINYXML2_LINK tinyxml2::tinyxml2)
elseif(TARGET tinyxml2)
  set(PACSIM_TINYXML2_LINK tinyxml2)
endif()

if(PACSIM_BUILD_ROS)
  find_package(ament_cmake_auto REQUIRED)
  find_package(ament_cmake REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(rclcpp_components REQUIRED)

  rosidl_generate_interfaces(
    ${PROJECT_NAME}
    "msg/StampedScalar.msg"
    "msg/Wheels.msg"
    "msg/Landmark.msg"
    "msg/PerceptionDetections.msg"
    "msg/GNSS.msg"
    "msg/Track.msg"
    "srv/ClockTriggerAbsolute.srv"
    "srv/ClockTriggerRelative.srv"
    DEPENDENCIES
     geometry_msgs
     std_msgs
     sensor_msgs
  )
endif()

file(GLOB_RECURSE pacsim_SRCS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.c"
        )

list(REMOVE_ITEM pacsim_SRCS "${PROJECT_SOURCE_DIR}/src/pacsim_main.cpp")
list(REMOVE_ITEM pacsim_SRCS "${PACSIM_PYTHON_BINDINGS_SOURCE}")

if(NOT PACSIM_BUILD_ROS)
  list(REMOVE_ITEM pacsim_SRCS "${PROJECT_SOURCE_DIR}/src/ros2Helpers.cpp")
endif()

list(APPEND pacsim_SRCS "${PROJECT_SOURCE_DIR}/third_party/stb/stb_image.cpp")

if(PACSIM_BUILD_ROS)
  add_library(pacsim_lib SHARED ${pacsim_SRCS})
else()
  add_library(pacsim_lib STATIC ${pacsim_SRCS})
  set_target_properties(pacsim_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(pacsim_lib PUBLIC PACSIM_STANDALONE=1)
endif()

target_include_directories(pacsim_lib PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb>"
  "$<INSTALL_INTERFACE:include>")

target_link_libraries(pacsim_lib 
  ${PACSIM_YAML_CPP_LINK}
  ${PACSIM_ASSIMP_LINK}
  Vulkan::Vulkan
  ${PACSIM_SHADERC_LINK}
  ${PACSIM_TINYXML2_LINK}
  ${CMAKE_DL_LIBS})

if(PACSIM_FORCE_SWIFTSHADER)
  target_compile_definitions(pacsim_lib PRIVATE PACSIM_FORCE_SWIFTSHADER=1)
endif()

if(PACSIM_SWIFTSHADER_ICD_JSON)
  if(NOT EXISTS "${PACSIM_SWIFTSHADER_ICD_JSON}")
    message(FATAL_ERROR "PACSIM_SWIFTSHADER_ICD_JSON does not exist: ${PACSIM_SWIFTSHADER_ICD_JSON}")
  endif()
  file(TO_CMAKE_PATH "${PACSIM_SWIFTSHADER_ICD_JSON}" PACSIM_SWIFTSHADER_ICD_JSON_CMAKE)
  target_compile_definitions(pacsim_lib PRIVATE
    PACSIM_SWIFTSHADER_ICD_JSON="${PACSIM_SWIFTSHADER_ICD_JSON_CMAKE}")

  if(PACSIM_BUNDLE_SWIFTSHADER)
    get_filename_component(PACSIM_SWIFTSHADER_DIR "${PACSIM_SWIFTSHADER_ICD_JSON}" DIRECTORY)
    file(GLOB PACSIM_SWIFTSHADER_RUNTIME_FILES CONFIGURE_DEPENDS
      "${PACSIM_SWIFTSHADER_DIR}/vk_swiftshader_icd.json"
      "${PACSIM_SWIFTSHADER_DIR}/vk_swiftshader.dll"
      "${PACSIM_SWIFTSHADER_DIR}/libvk_swiftshader.dll"
      "${PACSIM_SWIFTSHADER_DIR}/libvk_swiftshader.so"
      "${PACSIM_SWIFTSHADER_DIR}/libvk_swiftshader.so.*"
      "${PACSIM_SWIFTSHADER_DIR}/libvk_swiftshader.dylib"
      "${PACSIM_SWIFTSHADER_DIR}/vulkan-1.dll"
      "${PACSIM_SWIFTSHADER_DIR}/libvulkan.so"
      "${PACSIM_SWIFTSHADER_DIR}/libvulkan.so.*"
      "${PACSIM_SWIFTSHADER_DIR}/libvulkan.dylib"
      "${PACSIM_SWIFTSHADER_DIR}/libvulkan*.dylib")
    file(GLOB PACSIM_SWIFTSHADER_NOTICE_FILES CONFIGURE_DEPENDS
      "${PACSIM_SWIFTSHADER_DIR}/LICENSE"
      "${PACSIM_SWIFTSHADER_DIR}/LICENSE.txt"
      "${PACSIM_SWIFTSHADER_DIR}/AUTHORS"
      "${PACSIM_SWIFTSHADER_DIR}/AUTHORS.txt"
      "${PACSIM_SWIFTSHADER_DIR}/CONTRIBUTORS"
      "${PACSIM_SWIFTSHADER_DIR}/CONTRIBUTORS.txt"
      "${PACSIM_SWIFTSHADER_DIR}/NOTICE"
      "${PACSIM_SWIFTSHADER_DIR}/NOTICE.txt"
      "${PACSIM_SWIFTSHADER_DIR}/THIRD_PARTY_NOTICES"
      "${PACSIM_SWIFTSHADER_DIR}/THIRD_PARTY_NOTICES.txt")

    function(pacsim_copy_swiftshader_runtime target_name)
      if(PACSIM_SWIFTSHADER_RUNTIME_FILES)
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target_name}>/swiftshader"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PACSIM_SWIFTSHADER_RUNTIME_FILES}
            "$<TARGET_FILE_DIR:${target_name}>/swiftshader")
      endif()
    endfunction()
  endif()
endif()

if(PACSIM_BUILD_ROS)
  ament_target_dependencies(pacsim_lib 
    rclcpp std_srvs visualization_msgs geometry_msgs tf2_ros yaml-cpp)

  rosidl_target_interfaces(pacsim_lib
    ${PROJECT_NAME} "rosidl_typesupport_cpp")

  add_executable(${PROJECT_NAME}_node "${PROJECT_SOURCE_DIR}/src/pacsim_main.cpp")

  target_include_directories(${PROJECT_NAME}_node PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

  target_link_libraries(${PROJECT_NAME}_node 
    pacsim_lib 
    ${YAML_CPP_LIBRARIES}
  )

  ament_target_dependencies(${PROJECT_NAME}_node 
    rclcpp std_srvs visualization_msgs geometry_msgs tf2_ros yaml-cpp)

  rosidl_target_interfaces(${PROJECT_NAME}_node
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
endif()

if(PACSIM_BUILD_PYTHON)
  pybind11_add_module(pacsim_pybind "${PACSIM_PYTHON_BINDINGS_SOURCE}")
  if(NOT PACSIM_BUILD_ROS)
    target_compile_definitions(pacsim_pybind PRIVATE PACSIM_STANDALONE=1)
  endif()

  target_link_libraries(pacsim_pybind PUBLIC pacsim_lib Eigen3::Eigen)

  target_include_directories(pacsim_pybind PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb>"
    "$<INSTALL_INTERFACE:include>")
endif()

if(PACSIM_BUILD_PYTHON AND PACSIM_SWIFTSHADER_RUNTIME_FILES)
  pacsim_copy_swiftshader_runtime(pacsim_pybind)
  if(PACSIM_BUILD_ROS)
    pacsim_copy_swiftshader_runtime(pacsim_lib)
  endif()
endif()

if(PACSIM_BUILD_ROS)
  install(TARGETS
    ${PROJECT_NAME}_node
    pacsim_lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    ARCHIVE DESTINATION lib/${PROJECT_NAME}
  )

  if(PACSIM_BUILD_PYTHON)
    install(TARGETS pacsim_pybind
      RUNTIME DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
      LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
    )
    if(PACSIM_SWIFTSHADER_RUNTIME_FILES)
      install(FILES ${PACSIM_SWIFTSHADER_RUNTIME_FILES}
        DESTINATION lib/${PROJECT_NAME}/swiftshader)
      install(FILES ${PACSIM_SWIFTSHADER_RUNTIME_FILES}
        DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/swiftshader)
      if(PACSIM_SWIFTSHADER_NOTICE_FILES)
        install(FILES ${PACSIM_SWIFTSHADER_NOTICE_FILES}
          DESTINATION lib/${PROJECT_NAME}/swiftshader)
        install(FILES ${PACSIM_SWIFTSHADER_NOTICE_FILES}
          DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/swiftshader)
      endif()
    endif()
  endif()
else()
  if(PACSIM_BUILD_PYTHON)
    install(FILES
      "${PACSIM_PYTHON_SOURCE_DIR}/pacsim/__init__.py"
      "${PACSIM_PYTHON_SOURCE_DIR}/pacsim/_runtime.py"
      DESTINATION "${PACSIM_PYTHON_PACKAGE}")
    install(FILES
      "${PACSIM_PYTHON_SOURCE_DIR}/pacsim_pybind.py"
      DESTINATION ".")
    install(TARGETS pacsim_pybind
      RUNTIME DESTINATION "${PACSIM_PYTHON_PACKAGE}"
      LIBRARY DESTINATION "${PACSIM_PYTHON_PACKAGE}")
    if(PACSIM_SWIFTSHADER_RUNTIME_FILES)
      install(FILES ${PACSIM_SWIFTSHADER_RUNTIME_FILES}
        DESTINATION "${PACSIM_PYTHON_PACKAGE}/swiftshader")
      if(PACSIM_SWIFTSHADER_NOTICE_FILES)
        install(FILES ${PACSIM_SWIFTSHADER_NOTICE_FILES}
          DESTINATION "${PACSIM_PYTHON_PACKAGE}/swiftshader")
      endif()
    endif()
  endif()
endif()

file(GLOB_RECURSE pacsim_TRACKS "${PROJECT_SOURCE_DIR}/tracks/*.yaml")
file(GLOB_RECURSE pacsim_CONFIGS "${PROJECT_SOURCE_DIR}/config/*.yaml")
file(GLOB_RECURSE modells "${PROJECT_SOURCE_DIR}/urdf/*")

if(PACSIM_BUILD_ROS)
  install(FILES ${modells} DESTINATION share/${PROJECT_NAME}/urdf)
  install(FILES ${pacsim_TRACKS} DESTINATION tracks/${PROJECT_NAME})
  install(FILES ${pacsim_CONFIGS} DESTINATION config/${PROJECT_NAME})
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/assets/"
          DESTINATION "share/${PROJECT_NAME}/assets")

  ament_export_dependencies(std_msgs geometry_msgs rosidl_default_runtime)

  ament_auto_package(INSTALL_TO_SHARE
    tracks
    config
    launch
  )
endif()
