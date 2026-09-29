FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy \
        python3-yaml \
        python3-scipy \
        python3-colcon-common-extensions \
        gcc \
        ros-humble-rclpy \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
        ros-humble-visualization-msgs \
        ros-humble-rosbag2 \
        ros-humble-rosbag2-py \
        ros-humble-ros2bag \
        ros-humble-rosbag2-storage-default-plugins \
        ros-humble-launch \
        ros-humble-launch-ros \
    && rm -rf /var/lib/apt/lists/*

COPY src/obstacle_detector /ws/src/obstacle_detector
RUN gcc -O3 -fPIC -shared -o /usr/local/lib/libobstacle_components.so \
    /ws/src/obstacle_detector/native/components.c

COPY docker/obstacle-demo /usr/local/bin/obstacle-demo
COPY docker/obstacle-offline /usr/local/bin/obstacle-offline
COPY docker/obstacle-ros /usr/local/bin/obstacle-ros
COPY docker/live_view.py /usr/local/bin/live_view.py

WORKDIR /ws
RUN bash -lc "source /opt/ros/humble/setup.bash && colcon build --merge-install --packages-select obstacle_detector" \
    && chmod +x /usr/local/bin/obstacle-demo /usr/local/bin/obstacle-offline /usr/local/bin/obstacle-ros

ENV PYTHONPATH=/ws/install/lib/python3.10/site-packages
ENTRYPOINT ["/usr/local/bin/obstacle-offline"]
CMD ["--help"]
