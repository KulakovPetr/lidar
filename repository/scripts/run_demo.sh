#!/bin/bash
# Ubuntu demonstration: detector in Docker, standard ros2 bag play, local RViz2 when VIEW=rviz.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/demo_settings.sh" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HERE/demo_settings.sh"
  set +a
fi
if [ -z "${BAG:-}" ]; then
  echo "ОШИБКА: откройте demo_settings.sh и укажите каталог BAG с metadata.yaml."
  exit 1
fi
OUT="${OUT:?ОШИБКА: задайте пустой OUT}"
if [ ! -d "$BAG" ] || [ ! -f "$BAG/metadata.yaml" ]; then
  echo "ОШИБКА: BAG должен быть каталогом rosbag2 с metadata.yaml."
  exit 1
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
export NODE_NAME="${NODE_NAME:-obstacle-node}"
export VIEW="${VIEW:-rviz}"
export PORT="${PORT:-8091}"
export VIZ_HZ="${VIZ_HZ:-10}"
export RATE="${RATE:-1.0}"
export READ_AHEAD="${READ_AHEAD:-2}"
export WORKERS="${WORKERS:-2}"
export TOPIC="${TOPIC:-/lidar_points}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-120}"
export ROS_LOCALHOST_ONLY=1
completed=0
cleanup() {
  if [ "$completed" != "1" ]; then
    docker stop "$NODE_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
if [ "$VIEW" = "rviz" ]; then
  if ! command -v rviz2 >/dev/null 2>&1; then
    echo "RViz2 в этой среде не найден. Проверка RViz: НЕ ВЫПОЛНЕНА, это не PASS."
    echo "Предусловия Ubuntu 22.04:"
    echo "  sudo apt update"
    echo "  sudo apt install -y software-properties-common curl gnupg"
    echo "  sudo add-apt-repository universe"
    echo "  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg"
    echo "  echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main\" | sudo tee /etc/apt/sources.list.d/ros2.list"
    echo "  sudo apt update"
    echo "  sudo apt install -y ros-humble-desktop"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  export ROS_LOCALHOST_ONLY=1"
    echo "  export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
    echo "После установки повторите этот скрипт. Для записи без RViz задайте VIEW=web или VIEW=none."
    exit 2
  fi
fi
bash "$HERE/run_node.sh"
ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do
  if grep -q subscription_ready "$OUT/node_stderr.txt" "$OUT/node_stdout.txt" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" != "1" ]; then
  echo "ОШИБКА: подписка не стала готова за 40 с."
  exit 1
fi
echo "Подписка готова. Источник картинки — живые сообщения ROS."
if [ "$VIEW" = "web" ]; then
  echo "Откройте http://127.0.0.1:${PORT}/"
elif [ "$VIEW" = "rviz" ]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  rviz2 -d "$HERE/../src/obstacle_detector/rviz/detector.rviz" >/tmp/obstacle_rviz.txt 2>&1 &
  echo $! > "$OUT/rviz.pid"
  echo "RViz2 запущен локально. Конфигурация: src/obstacle_detector/rviz/detector.rviz"
fi
if [ "${DEMO_NO_PAUSE:-0}" != "1" ]; then
  echo "Включите запись экрана и нажмите Enter."
  read -r _
fi
echo "Скорость плеера RATE=${RATE}. Это не частота картинки."
echo "Предел новых кадров визуализации: ${VIZ_HZ} Гц."
export PLAY_SECONDS="${PLAY_SECONDS:-}"
export START_OFFSET="${START_OFFSET:-}"
bash "$HERE/run_play.sh"
launch_pid="$(tr -d '[:space:]' < "$OUT/launch.pid")"
docker exec -u "$(id -u):$(id -g)" "$NODE_NAME" kill -INT "$launch_pid" || true
flushed=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
  if [ -f "$OUT/node_counts.json" ] && grep -q '"journal_closed": true' "$OUT/node_counts.json"; then
    flushed=1
    break
  fi
  sleep 1
done
if [ "$flushed" != "1" ]; then
  echo "ОШИБКА: журнал не подтвердил закрытие."
  exit 1
fi
echo "Счётчики: $OUT/node_counts.json"
echo "Просмотр остаётся, пока контейнер жив. Закрытие: docker stop ${NODE_NAME}"
completed=1
