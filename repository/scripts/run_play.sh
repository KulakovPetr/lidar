#!/bin/bash
# Standard ros2 bag play on the host network. Not the detector.
set -eu
BAG="${BAG:?ОШИБКА: задайте BAG}"
RATE="${RATE:-1.0}"
PLAY_SECONDS="${PLAY_SECONDS:-}"
DOMAIN="${ROS_DOMAIN_ID:-120}"
IMAGE="${IMAGE:-lidar-obstacle:acceptance}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TOPIC="${TOPIC:-/lidar_points}"
echo "ros2 bag play, топик ${TOPIC}, скорость плеера ${RATE}. Это не частота обработки."
docker run --rm --network host --shm-size 512m --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID="$DOMAIN" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml \
  -e TOPIC="$TOPIC" -e RATE="$RATE" -e PLAY_SECONDS="$PLAY_SECONDS" \
  -e READ_AHEAD="${READ_AHEAD:-2}" -e START_OFFSET="${START_OFFSET:-}" \
  -v "$BAG:/bag:ro" \
  -v "$HERE/play_inside.sh:/play_inside.sh:ro" \
  -v "$HERE/fastdds.xml:/fastdds.xml:ro" \
  --entrypoint bash "$IMAGE" /play_inside.sh
echo "Публикация bag закончилась. Нода досчитает не больше одного ожидающего облака."
