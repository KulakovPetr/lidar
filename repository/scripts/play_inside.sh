#!/bin/bash
# Standard ros2 bag play. Start the subscriber first.
# The installed player defaults to a read-ahead of 1000. These clouds are about
# 5 MB, so that default fills for a long time and publishes nothing.
set +u
source /opt/ros/humble/setup.bash
set +u
source /ws/install/setup.bash
set +u
TOPIC="${TOPIC:-/lidar_points}"
RATE="${RATE:-1.0}"
READ_AHEAD="${READ_AHEAD:-2}"
OFFSET_ARG=""
if [ -n "${START_OFFSET:-}" ]; then
  OFFSET_ARG="--start-offset ${START_OFFSET}"
fi
echo "PLAY_TOPIC=${TOPIC} RATE=${RATE} READ_AHEAD=${READ_AHEAD} START_OFFSET=${START_OFFSET:-0} DOMAIN=${ROS_DOMAIN_ID:-} LOCALHOST=${ROS_LOCALHOST_ONLY:-}"
echo "COMMAND ros2 bag play /bag -r ${RATE} --disable-keyboard-controls --read-ahead-queue-size ${READ_AHEAD} ${OFFSET_ARG}"
ros2 bag play /bag -r "${RATE}" --disable-keyboard-controls --read-ahead-queue-size "${READ_AHEAD}" ${OFFSET_ARG} > /tmp/play_stdout.txt 2> /tmp/play_stderr.txt &
play_pid=$!
echo "PLAYER_PID=${play_pid}"
if [ -n "${PLAY_SECONDS}" ]; then
  sleep "${PLAY_SECONDS}"
  kill "${play_pid}" 2>/dev/null || true
  sleep 1
  if kill -0 "${play_pid}" 2>/dev/null; then
    pkill -KILL -P "${play_pid}" 2>/dev/null || true
    kill -9 "${play_pid}" 2>/dev/null || true
  fi
  wait "${play_pid}" 2>/dev/null || true
  echo "PLAY_STDOUT"
  cat /tmp/play_stdout.txt || true
  echo "PLAY_STDERR"
  cat /tmp/play_stderr.txt || true
  exit 0
fi
wait "${play_pid}"
code=$?
echo "PLAY_STDOUT"
cat /tmp/play_stdout.txt || true
echo "PLAY_STDERR"
cat /tmp/play_stderr.txt || true
exit "${code}"
