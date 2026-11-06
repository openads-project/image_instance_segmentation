cmake_minimum_required(VERSION 3.8)
project(image_instance_segmentation LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc video)
find_package(image_transport REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(triton_cpp REQUIRED)
find_package(vision_msgs REQUIRED)

add_library(${PROJECT_NAME}_lib SHARED
  src/ImageInstanceSegmentation.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_compile_features(${PROJECT_NAME}_lib PUBLIC cxx_std_17)

ament_target_dependencies(${PROJECT_NAME}_lib
  image_transport
  rcl_interfaces
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  triton_cpp
  vision_msgs
)
target_link_libraries(${PROJECT_NAME}_lib ${OpenCV_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "image_instance_segmentation::ImageInstanceSegmentation"
  EXECUTABLE ${PROJECT_NAME}
)

set(_triton_runtime_paths "/opt/tritonclient/lib;/opt/tritonclient/tritonserver/clients/lib")
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  BUILD_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH "${_triton_runtime_paths}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  # Preserve the existing lint scope; clang-tidy adoption is a separate refactor.
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_clang_tidy)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_clang_format_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/format/.clang-format)
  set(ament_cmake_clang_tidy_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/.clang-tidy)
  set(ament_cmake_flake8_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/ament_flake8.ini)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
