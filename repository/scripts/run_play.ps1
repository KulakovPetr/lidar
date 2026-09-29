# Стандартный ros2 bag play в том же контейнере, что и нода.
# Сначала должна быть запущена нода. Плеер сам bag не подменяет.
# Установленный Humble по умолчанию читает 1000 сообщений вперёд. Для этих облаков
# это слишком много, поэтому play_inside.sh задаёт --read-ahead-queue-size 2.
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
function Fail([string]$Message) { [Console]::Out.WriteLine($Message); exit 1 }
if (-not $env:BAG) { Fail 'ОШИБКА: задайте BAG.' }
if (-not (Test-Path -LiteralPath $env:BAG)) { Fail ('ОШИБКА: каталог BAG не найден: ' + $env:BAG) }
$Rate = if ($env:RATE) { $env:RATE } else { '1.0' }
$ReadAhead = if ($env:READ_AHEAD) { $env:READ_AHEAD } else { '2' }
$StartOffset = if ($env:START_OFFSET) { $env:START_OFFSET } else { '' }
$PlaySeconds = if ($env:PLAY_SECONDS) { $env:PLAY_SECONDS } else { '' }
$Domain = if ($env:ROS_DOMAIN_ID) { $env:ROS_DOMAIN_ID } else { '120' }
$Name = if ($env:NODE_NAME) { $env:NODE_NAME } else { 'obstacle-node' }
[Console]::Out.WriteLine("ros2 bag play, скорость плеера $Rate, read-ahead $ReadAhead. Это не частота обработки и не 10 Гц.")
docker exec $Name test -d /bag
if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: в контейнере ноды нет /bag. Запустите run_node.ps1 с тем же BAG.' }
$Topic = if ($env:TOPIC) { $env:TOPIC } else { '/lidar_points' }
docker exec -u 1000:1000 -e HOME=/tmp -e "ROS_DOMAIN_ID=$Domain" -e ROS_LOCALHOST_ONLY=1 -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml -e "TOPIC=$Topic" -e "RATE=$Rate" -e "READ_AHEAD=$ReadAhead" -e "PLAY_SECONDS=$PlaySeconds" -e "START_OFFSET=$StartOffset" $Name bash /play_inside.sh
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 137 -and $LASTEXITCODE -ne 143) { Fail ('ОШИБКА: ros2 bag play завершился с кодом ' + $LASTEXITCODE) }
[Console]::Out.WriteLine('Публикация bag закончилась. Нода ещё может досчитать текущий кадр. Очередь приложения хранит одно облако.')
exit 0
