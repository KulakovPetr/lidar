"""Standard ros2 bag play. This process is not the detector."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bag = LaunchConfiguration("bag")
    rate = LaunchConfiguration("rate")
    return LaunchDescription(
        [
            DeclareLaunchArgument("bag"),
            DeclareLaunchArgument("rate", default_value="1.0"),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "play",
                    bag,
                    "-r",
                    rate,
                    "--disable-keyboard-controls",
                ],
                output="screen",
            ),
        ]
    )
