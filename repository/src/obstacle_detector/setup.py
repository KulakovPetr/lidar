from setuptools import find_packages, setup

package_name = "obstacle_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", [
            "config/pipeline.yaml",
            "config/detector.yaml",
            "config/corridor.yaml",
            "config/batch.yaml",
            "config/configured_straight.yaml",
            "config/mount_unconfirmed.yaml",
            "config/session.yaml",
            "config/intrusion_rule.yaml",
        ]),
        ("share/" + package_name + "/launch", [
            "launch/detector.launch.py",
            "launch/play.launch.py",
            "launch/rviz.launch.py",
        ]),
        ("share/" + package_name + "/rviz", ["rviz/detector.rviz", "rviz/detector_livox.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="obstacle-detector",
    maintainer_email="unset@localhost",
    description="Contracts and geometry checks for a metro-train lidar.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "obstacle_node = obstacle_detector.ros_node:main",
            "sequential_play = obstacle_detector.sequential_play:main",
        ],
    },
)
