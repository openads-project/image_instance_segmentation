#!/usr/bin/env python3

# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from tracetools_launch.action import Trace


def generate_launch_description():
    """Generate the image instance segmentation launch description."""
    remappable_topics = [
        DeclareLaunchArgument("diagnostics_topic", default_value="/diagnostics", description="diagnostics topic remap"),
        DeclareLaunchArgument("image_topic", default_value="~/image", description="input image topic remap"),
        DeclareLaunchArgument(
            "class_mask_topic", default_value="~/segmentation/class_mask", description="semantic mask topic remap"
        ),
        DeclareLaunchArgument(
            "instance_mask_topic", default_value="~/segmentation/instance_mask", description="instance mask topic remap"
        ),
        DeclareLaunchArgument(
            "detections_topic", default_value="~/segmentation/detections", description="2D detections topic remap"
        ),
        DeclareLaunchArgument(
            "road_users_topic", default_value="~/perception/road_users", description="road-user objects topic remap"
        ),
        DeclareLaunchArgument(
            "traffic_lights_topic", default_value="~/perception/traffic_lights", description="traffic-light objects topic remap"
        ),
        DeclareLaunchArgument(
            "traffic_signs_topic", default_value="~/perception/traffic_signs", description="traffic-sign objects topic remap"
        ),
        DeclareLaunchArgument(
            "class_overlay_topic", default_value="~/debug/overlay/class", description="class overlay topic remap"
        ),
        DeclareLaunchArgument(
            "instance_overlay_topic",
            default_value="~/debug/overlay/instance",
            description="instance overlay topic remap",
        ),
    ]
    args = [
        DeclareLaunchArgument("name", default_value="image_instance_segmentation", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "params",
            default_value=os.path.join(
                get_package_share_directory("image_instance_segmentation"), "config", "params.yml"
            ),
            description="path to parameter file",
        ),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="use simulation clock"),
        DeclareLaunchArgument("ros_tracing", default_value="false", description="Enable tracing"),
        DeclareLaunchArgument("triton_server_url", default_value="triton-server:8001", description="Triton server URL"),
        *remappable_topics,
    ]

    nodes = [
        Node(
            package="image_instance_segmentation",
            executable="image_instance_segmentation",
            name=LaunchConfiguration("name"),
            namespace=LaunchConfiguration("namespace"),
            parameters=[
                LaunchConfiguration("params"),
                {"prediction.triton_server_url": LaunchConfiguration("triton_server_url")},
            ],
            arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            remappings=[
                (argument.default_value[0].text, LaunchConfiguration(argument.name)) for argument in remappable_topics
            ],
            output="screen",
            emulate_tty=True,
        )
    ]

    if Trace is not None:
        nodes.append(
            Trace(
                session_name="trace",
                dual_session=True,
                condition=IfCondition(LaunchConfiguration("ros_tracing")),
            )
        )

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            *nodes,
        ]
    )
