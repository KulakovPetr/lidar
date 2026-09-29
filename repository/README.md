# Исходники детектора

Пакет ROS 2 `obstacle_detector` читает `sensor_msgs/PointCloud2`, считает кандидатов в прямоугольном профиле и публикует компактный результат. Этот каталог — снимок исходников, конфигурации, Docker-файлов и проверенных скриптов запуска.

Руководство: [../documentation/User_guide.docx](../documentation/User_guide.docx). Видео: [../prototype/Video_ROS.mp4](../prototype/Video_ROS.mp4). Презентация: [../presentation/Presentation.pptx](../presentation/Presentation.pptx).

## Что входит

- `src/obstacle_detector` — пакет, конфигурация `config/`, запуск `launch/detector.launch.py`, проверки `test/`.
- `docker/Dockerfile.acceptance` — слой проверенного образа поверх локальной базы.
- `docker/Dockerfile.stage4` и `docker/Dockerfile.core-ros` — более ранние ступени. Цепочка до публичной базы не замкнута, см. ниже.
- `docker/live_view.py` — страница просмотра. Она подписывается на уже опубликованные сообщения и детектор не пересчитывает.
- `scripts/` — запуск ноды, стандартный `ros2 bag play`, остановка и RViz.

В каталог не входят bag, распакованные облака, архив образа, журналы прогонов и кэши.

## Требования

- Ubuntu 22.04 и ROS 2 Humble — целевая среда организатора.
- Docker. На машине подготовки проверен Docker Desktop на Windows 10.
- Вход: `sensor_msgs/PointCloud2` из `ros2 bag play`.
- Память контейнера в скриптах: 6 ГБ, общая память 512 МБ. Два таких контейнера одновременно не запускать.
- Вычисление: CPU, `workers=2`. GPU в рабочем тракте не используется. Значение `gpu` в параметре остаётся на CPU и пишет ошибку в журнал.

## Сборка

Проверенный образ:

`sha256:0c277fdc3b756013a1ee5f3d0b36c1c12a1caf312a18d5db174f2d81bf1253b1`

Локальный тег `lidar-obstacle:acceptance` на этот id указывает, но тег версией не является.

`docker/Dockerfile.acceptance` начинается со строки `FROM lidar-obstacle:stage4-demo`. Этот базовый образ есть только на машине подготовки:

`sha256:38328b243184faa6403091f88fa366a37f2589858ffae3fbde8f3b5084ebb012`

`docker/Dockerfile.stage4` в свою очередь начинается с `lidar-obstacle:stage3-intrusion`. Dockerfile этой ступени в комплекте нет. `docker/Dockerfile.core-ros` начинается с публичного `ros:humble-ros-base-jammy`, но сам по себе текущий рабочий образ не собирает.

На этой машине скрипты запускают локальный тег `lidar-obstacle:acceptance`. Он уже есть и контейнер с ним отвечает на http://127.0.0.1:8091/ .

Публичный пакет собирает GitHub Actions из `docker/Dockerfile` и публикует `ghcr.io/kulakovpetr/lidar/obstacle-detector:latest`. Основа — `ros:humble-ros-base-jammy`. Это тот же способ, что в cinemaabyss-homework, и это не побайтовая копия локального `acceptance`: слой `Dockerfile.acceptance` начинается с локального `lidar-obstacle:stage4-demo`, которого в GitHub нет.

После успешной сборки Actions:

```bash
docker pull ghcr.io/kulakovpetr/lidar/obstacle-detector:latest
docker tag ghcr.io/kulakovpetr/lidar/obstacle-detector:latest lidar-obstacle:acceptance
```

Откатные локальные образы удалять нельзя:

- `lidar-obstacle:rollback-09e7182b` — `sha256:bdf7d56cbfc57994a1b49976dc6d8383164921f5d8db46e439743ab75035f61a`
- `lidar-obstacle:rollback-8b1ecb6a` — `sha256:3cb08ac2e493197319ddde0898d3cd048833e1565fb50793b3d1835d3ee50687`
- `lidar-obstacle:rollback-7889f7d0` — `sha256:c80f34cde8fedf8fcb67e78bcc6bf0103235be7f37194758a5ced44eafe0c237`

## Запуск

Команды ниже выполняются из этого каталога `repository/`. Скрипты сами передают профиль `configured_straight`, два worker, частоту визуализации и очередь чтения плеера. Не меняйте геометрию и пороги в yaml ради демонстрации.

Оба поддерживаемых топика:

- `/lidar_points`, кадр `hesai_lidar`;
- `/sensing/lidar/hesai128/pointcloud`, кадр `lidar_livox`.

По умолчанию берётся первый. Второй задаётся переменной `TOPIC`.

Windows. Сначала проверьте путь записи в `scripts/demo_settings.ps1`, затем из этого каталога:

```powershell
powershell -File scripts/run_demo.ps1
```

Ubuntu:

```bash
BAG=<каталог rosbag2> OUT=<новый пустой каталог> VIEW=web bash scripts/run_demo.sh
```

`run_demo` поднимает ноду и затем вызывает стандартный плеер:

`ros2 bag play <bag> -r 1.0 --disable-keyboard-controls --read-ahead-queue-size 2`

`RATE` — скорость плеера, не скорость обработки. `VIZ_HZ` — предел новых картинок в секунду. Значения 1, 2, 5 и 10 принимаются; принимается и другое положительное число не больше 60. Ноль и нечисло отвергаются. Экономичный просмотр — 2 Гц, запись видео — 10 Гц. Если нового результата нет, облако заново не строится.

`PLAY_SECONDS`, если задан, ограничивает время работы плеера по часам, а не диапазон кадров bag.

Страница просмотра на Windows: `http://127.0.0.1:8091/`. На Ubuntu при `VIEW=web` контейнер в сети host, страница на порту 8080. `VIEW=rviz` на Ubuntu запускает `rviz2` на хосте. Если команды нет, скрипт печатает установку ROS Humble и завершается с кодом 2. На машине подготовки RViz не открывался.

Выходной каталог, если он задан и пуст:

- `frames.jsonl` — полные результаты;
- `publish_timing.jsonl`, `viz_timing.jsonl`;
- `node_counts.json` после корректной остановки;
- `run.lock` на время работы.

Остановка: `scripts/stop_node.sh` внутри контейнера посылает SIGINT процессу ноды, чтобы журнал закрылся. На Windows после просмотра: `docker stop obstacle-node`. Имя контейнера задаёт `NODE_NAME`, по умолчанию `obstacle-node`.

## Проверки в пакете

`src/obstacle_detector/test` — небольшие проверки контракта и правил. Они рассчитаны на Python с `rclpy` внутри образа, не на произвольный Python хоста. Полный bag ради этой упаковки повторно не прогонялся. Параметры запуска: [../additional-parameters/README.md](../additional-parameters/README.md).
