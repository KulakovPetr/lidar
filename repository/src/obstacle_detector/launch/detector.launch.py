from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/lidar_points"),
            DeclareLaunchArgument("corridor_source", default_value="configured_straight"),
            DeclareLaunchArgument("profile", default_value=""),
            DeclareLaunchArgument("mount", default_value=""),
            DeclareLaunchArgument("frame_budget_s", default_value="30.0"),
            DeclareLaunchArgument("run_detector", default_value="true"),
            DeclareLaunchArgument("budget_is_artificial_timeout", default_value="false"),
            DeclareLaunchArgument("playback_mode", default_value="stream"),
            DeclareLaunchArgument("output_dir", default_value="/output"),
            DeclareLaunchArgument("compute_backend", default_value="cpu"),
            DeclareLaunchArgument("queue_policy", default_value="keep_latest"),
            DeclareLaunchArgument("subscription_depth", default_value="10"),
            DeclareLaunchArgument("publish_visualization", default_value="true"),
            DeclareLaunchArgument("visualization_enabled", default_value="true"),
            DeclareLaunchArgument("visualization_rate_hz", default_value="2.0"),
            DeclareLaunchArgument("display_background_limit", default_value="3500"),
            DeclareLaunchArgument("workers", default_value="2"),
            Node(
                package="obstacle_detector",
                executable="obstacle_node",
                name="obstacle_detector",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "corridor_source": LaunchConfiguration("corridor_source"),
                        "profile": LaunchConfiguration("profile"),
                        "mount": LaunchConfiguration("mount"),
                        "frame_budget_s": LaunchConfiguration("frame_budget_s"),
                        "run_detector": LaunchConfiguration("run_detector"),
                        "budget_is_artificial_timeout": LaunchConfiguration("budget_is_artificial_timeout"),
                        "playback_mode": LaunchConfiguration("playback_mode"),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "compute_backend": LaunchConfiguration("compute_backend"),
                        "queue_policy": LaunchConfiguration("queue_policy"),
                        "subscription_depth": LaunchConfiguration("subscription_depth"),
                        "publish_visualization": LaunchConfiguration("publish_visualization"),
                        "visualization_enabled": LaunchConfiguration("visualization_enabled"),
                        "visualization_rate_hz": LaunchConfiguration("visualization_rate_hz"),
                        "display_background_limit": LaunchConfiguration("display_background_limit"),
                        "workers": LaunchConfiguration("workers"),
                    }
                ],
            ),
        ]
    )
