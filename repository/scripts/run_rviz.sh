#!/bin/bash
# RViz2 on the Ubuntu host. This script does not start a container.
set -eu
TOPIC="${TOPIC:-/lidar_points}"
DOMAIN="${ROS_DOMAIN_ID:-120}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="$ROOT/src/obstacle_detector/rviz/detector.rviz"
if [ "$TOPIC" = "/sensing/lidar/hesai128/pointcloud" ]; then
  CONFIG="$ROOT/src/obstacle_detector/rviz/detector_livox.rviz"
fi
if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi
if ! command -v rviz2 >/dev/null 2>&1; then
  echo "Окно RViz не открыто: команда rviz2 не установлена." >&2
  echo "На Ubuntu 22.04 сначала подключите официальный репозиторий ROS 2, затем поставьте desktop:" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt install -y software-properties-common curl gnupg" >&2
  echo "  sudo add-apt-repository -y universe" >&2
  echo "  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg" >&2
  echo "  echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(. /etc/os-release && echo \$UBUNTU_CODENAME) main\" | sudo tee /etc/apt/sources.list.d/ros2.list" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt install -y ros-humble-desktop" >&2
  echo "  source /opt/ros/humble/setup.bash" >&2
  echo "  export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=${DOMAIN}" >&2
  echo "  bash repository/scripts/run_rviz.sh" >&2
  echo "Нода детектора при этом работает в Docker и публикует в ROS_DOMAIN_ID=${DOMAIN}." >&2
  echo "Fixed Frame в конфигурации: hesai_lidar, для топика Hesai — lidar_livox. Отдельного TF нет." >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "Файл конфигурации RViz не найден: ${CONFIG}" >&2
  exit 1
fi
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="$DOMAIN"
export FASTRTPS_DEFAULT_PROFILES_FILE="$HERE/fastdds.xml"
echo "Локальный rviz2. Fixed Frame берётся из конфигурации, не из DISPLAY."
exec rviz2 -d "$CONFIG"
