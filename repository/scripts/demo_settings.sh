# Единственное место настроек на Ubuntu. Файл читает run_demo.sh.
# Подставьте каталог rosbag2, внутри которого есть metadata.yaml.
# Проверенный файл записи лежит на машине подготовки и в этот репозиторий не входит.

: "${BAG:=}"
: "${TOPIC:=/lidar_points}"
: "${VIEW:=web}"
: "${VIZ_HZ:=10}"
: "${RATE:=1.0}"
: "${READ_AHEAD:=2}"
: "${WORKERS:=2}"
: "${ROS_DOMAIN_ID:=120}"
# Пусто — играть всю запись. 45 — только начало по часам компьютера.
: "${PLAY_SECONDS:=}"
if [ -z "${OUT:-}" ]; then
  OUT="${HOME}/lidar-demo-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$OUT"
fi
