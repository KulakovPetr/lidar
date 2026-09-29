#!/bin/bash
# Keeps the chosen viewer after the detector has flushed its journal.
# The detector is still the same ros2 launch as obstacle-ros.
set +u
source /opt/ros/humble/setup.bash
set +u
source /ws/install/setup.bash
set +u
export AMENT_PREFIX_PATH="/ws/install:${AMENT_PREFIX_PATH:-}"
export PYTHONUNBUFFERED=1
VIEW="${VIEW:-web}"
ros2 launch obstacle_detector detector.launch.py "$@" > /output/node_stdout.txt 2> /output/node_stderr.txt &
launch_pid=$!
echo "$launch_pid" > /output/launch.pid
echo "LAUNCH_PID=${launch_pid}"
if [ "$VIEW" = "web" ]; then
  python3 /usr/local/bin/live_view.py > /output/view_stdout.txt 2> /output/view_stderr.txt &
  echo $! > /output/view.pid
  echo "VIEW_URL=http://127.0.0.1:${HOST_PORT:-8091}/"
fi
wait "$launch_pid"
echo "DETECTOR_STOPPED"
if [ "$VIEW" = "web" ] && [ -f /output/view.pid ]; then
  view_pid=$(cat /output/view.pid)
  while kill -0 "$view_pid" 2>/dev/null; do
    sleep 1
  done
fi
