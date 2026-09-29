"""RViz2 only. It does not start the detector or a bag."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    default = os.path.join(get_package_share_directory("obstacle_detector"), "rviz", "detector.rviz")
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default),
            ExecuteProcess(cmd=["rviz2", "-d", config], output="screen"),
        ]
    )
