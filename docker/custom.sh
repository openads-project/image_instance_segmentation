#!/bin/bash
# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

set -e

# Jazzy tracing with message-link instrumentation and dual-session support.
tracing_workspace="$(mktemp -d)"
trap 'rm -rf "$tracing_workspace"' EXIT

git clone --depth 1 --branch jazzy-ika https://github.com/RaphvK/ros2_tracing.git "$tracing_workspace/src/ros2_tracing"
apt-get update
rosdep update
rosdep install -y -i --from-paths "$tracing_workspace/src/ros2_tracing" \
    -t build -t buildtool -t build_export -t buildtool_export -t exec
source "/opt/ros/${ROS_DISTRO}/setup.bash"

colcon --log-base "$tracing_workspace/log" build \
    --base-paths "$tracing_workspace/src/ros2_tracing" \
    --packages-up-to tracetools tracetools_launch ros2trace \
    --build-base "$tracing_workspace/build" \
    --install-base /docker-ros/ws/install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    --allow-overriding tracetools tracetools_launch tracetools_trace ros2trace lttngpy
