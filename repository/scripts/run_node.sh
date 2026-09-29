#!/bin/bash
# Standalone detector. The bag is mounted for the player; the node does not read it.
set -eu
OUT="${OUT:?ОШИБКА: задайте OUT}"
BAG="${BAG:?ОШИБКА: задайте BAG}"
TOPIC="${TOPIC:-/lidar_points}"
BACKEND="${BACKEND:-cpu}"
BUDGET="${BUDGET:-30.0}"
DOMAIN="${ROS_DOMAIN_ID:-120}"
IMAGE="${IMAGE:-lidar-obstacle:acceptance}"
NAME="${NODE_NAME:-obstacle-node}"
WORKERS="${WORKERS:-2}"
VIZ_HZ="${VIZ_HZ:-2}"
VIZ_ENABLED="${VIZ_ENABLED:-true}"
VIEW="${VIEW:-none}"
PORT="${PORT:-8091}"
case "$WORKERS" in
  1|2|4|6) ;;
  *) echo "ОШИБКА: WORKERS должен быть 1, 2, 4 или 6."; exit 1 ;;
esac
case "$VIEW" in
  web|rviz|none) ;;
  *) echo "ОШИБКА: VIEW должен быть web, rviz или none."; exit 1 ;;
esac
if [ "$BACKEND" != "cpu" ]; then
  echo "ОШИБКА: эта поставка запускает только compute_backend=cpu."
  exit 1
fi
docker version >/dev/null
docker image inspect "$IMAGE" >/dev/null
mkdir -p "$OUT"
if [ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]; then
  echo "ОШИБКА: каталог OUT не пуст, журнал не дописывается: $OUT"
  exit 1
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# host network already publishes the viewer on port 8080. Do not add -p.
if [ "$VIEW" = "web" ]; then
  PORT=8080
fi
echo "Профиль: configured_straight, 2.1 x 3.0 м, s=1..120 м. Это допущение, не восстановленная траектория."
echo "Топик ${TOPIC}. Workers ${WORKERS}. Картинка enabled=${VIZ_ENABLED}, предел ${VIZ_HZ} Гц. Просмотр ${VIEW}."
docker run -d --name "$NAME" --network host --shm-size 512m --memory 6g --user "$(id -u):$(id -g)" \
  -e HOME=/output -e PYTHONUNBUFFERED=1 \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID="$DOMAIN" -e VIEW="$VIEW" -e HOST_PORT="$PORT" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml \
  -v "$OUT:/output" \
  -v "$BAG:/bag:ro" \
  -v "$HERE/demo_keep.sh:/demo_keep.sh:ro" \
  -v "$HERE/play_inside.sh:/play_inside.sh:ro" \
  -v "$HERE/stop_node.sh:/stop_node.sh:ro" \
  -v "$HERE/fastdds.xml:/fastdds.xml:ro" \
  --entrypoint bash "$IMAGE" /demo_keep.sh \
  "input_topic:=$TOPIC" corridor_source:=configured_straight \
  "frame_budget_s:=$BUDGET" playback_mode:=stream \
  "compute_backend:=$BACKEND" "workers:=$WORKERS" queue_policy:=keep_latest subscription_depth:=10 \
  publish_visualization:=true "visualization_enabled:=$VIZ_ENABLED" "visualization_rate_hz:='${VIZ_HZ}'" output_dir:=/output
echo "Нода ${NAME} запущена. Журнал ${OUT}"
echo "Закрытие демонстрации: docker stop ${NAME}"
