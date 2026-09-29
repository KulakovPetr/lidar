# Обнаружение препятствий по облаку точек лидара

Прототип читает стандартный поток ROS 2 `sensor_msgs/PointCloud2`, ищет поверхности в заданном прямоугольном профиле и публикует компактный результат с расстоянием от лидара. Это геометрический детектор. Он не управляет торможением.

На проверенном полном прогоне bag `cloud_with_fake_obj` получено 1510 сообщений, записано 1509 результатов, один кадр снят очередью приложения. Среднее за всё время скрипта около 3.12 результата в секунду. Дальность ближайшего условного вторжения на коротком срезе того же bag — около 98.7 м. Горизонт поиска 120 м этой дальностью не является.

## Разделы

- [repository/](repository/README.md) — исходники, конфигурация, Docker и скрипты запуска.
- [documentation/](documentation/User_guide.docx) — руководство пользователя.
- [presentation/](presentation/Presentation.pptx) — презентация.
- [prototype/](prototype/Video_ROS.mp4) — видео работы.
- [additional-parameters/](additional-parameters/README.md) — параметры запуска и образ.

Соответствие полям формы: [submission_index.md](submission_index.md). Что ещё не готово: [submission_checklist.md](submission_checklist.md).

## Быстрый старт

Настройки — в файле [repository/scripts/demo_settings.ps1](repository/scripts/demo_settings.ps1). Новая папка результата создаётся сама. Как устроена страница, написано в [documentation/User_guide.docx](documentation/User_guide.docx).

Windows, из корня этого репозитория:

```powershell
powershell -File repository/scripts/run_demo.ps1
```

Откроется [http://127.0.0.1:8091/](http://127.0.0.1:8091/). Нажмите Enter в окне PowerShell, когда будете готовы смотреть. Остановка: `docker stop obstacle-node`.

На Ubuntu сначала впишите свой каталог записи в `repository/scripts/demo_settings.sh`, затем `bash repository/scripts/run_demo.sh`.

## Версия

| Что | Значение |
| --- | --- |
| Образ, на котором выполнены проверки | `sha256:0c277fdc3b756013a1ee5f3d0b36c1c12a1caf312a18d5db174f2d81bf1253b1` |
| Алгоритм | `09e7182b2921aa78` |
| Конфигурация | `7ae8ad5bdc96372e` |
| Тег `lidar-obstacle:acceptance` | метка локального образа, не номер версии |

Контейнер публикуется так же, как в cinemaabyss-homework: GitHub Actions собирает `repository/docker/Dockerfile` и кладёт образ в `ghcr.io/kulakovpetr/lidar/obstacle-detector`. Тег `latest` — эта сборка от публичного `ros:humble-ros-base-jammy`. Локальный проверенный образ `lidar-obstacle:acceptance` (`sha256:0c277fdc…`) на GitHub Packages этой отправкой не заливается: у входа GitHub на машине нет права `packages`. Подробности — в [repository/README.md](repository/README.md).

## Ограничения

Прямой профиль — выбранное допущение, а не подтверждённая траектория. Расстояние считается от лидара. Расстояние от носа состава в коде не определено. Провода и все малые объекты отдельным классом не подтверждены. Десять герц визуализации — предел новых картинок, а не подтверждённая скорость детектора. Окно RViz на машине подготовки не открывалось.

## Где смотреть демонстрацию

Готовое видео: [prototype/Video_ROS.mp4](prototype/Video_ROS.mp4). Живая страница после запуска: [http://127.0.0.1:8091/](http://127.0.0.1:8091/). На ней кнопка «Экран записи».
