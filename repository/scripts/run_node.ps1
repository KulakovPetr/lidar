# Самостоятельная ROS-нода. Bag этим скриптом не проигрывается.
# UTF-8 с BOM для Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
function Fail([string]$Message) { [Console]::Out.WriteLine($Message); exit 1 }
if (-not $env:OUT) { Fail 'ОШИБКА: задайте OUT для журнала. Каталог должен быть пустым.' }
if (-not $env:BAG) { Fail 'ОШИБКА: задайте BAG. Нода сама bag не читает, но плеер в этом контейнере его монтирует.' }
$Topic = if ($env:TOPIC) { $env:TOPIC } else { '/lidar_points' }
$Backend = if ($env:BACKEND) { $env:BACKEND } else { 'cpu' }
$Budget = if ($env:BUDGET) { $env:BUDGET } else { '30.0' }
$Domain = if ($env:ROS_DOMAIN_ID) { $env:ROS_DOMAIN_ID } else { '120' }
$Image = if ($env:IMAGE) { $env:IMAGE } else { 'lidar-obstacle:acceptance' }
$Name = if ($env:NODE_NAME) { $env:NODE_NAME } else { 'obstacle-node' }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { '2' }
$VizHz = if ($env:VIZ_HZ) { $env:VIZ_HZ } else { '2' }
$VizEnabled = if ($env:VIZ_ENABLED) { $env:VIZ_ENABLED } else { 'true' }
$View = if ($env:VIEW) { $env:VIEW } else { 'none' }
$Port = if ($env:PORT) { $env:PORT } else { '8091' }
if ($Workers -notin @('1', '2', '4', '6')) { Fail 'ОШИБКА: WORKERS должен быть 1, 2, 4 или 6.' }
if ($View -notin @('web', 'rviz', 'none')) { Fail 'ОШИБКА: VIEW должен быть web, rviz или none.' }
if ($Backend -ne 'cpu') { Fail 'ОШИБКА: эта поставка запускает только compute_backend=cpu.' }
docker version | Out-Null
if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: Docker не отвечает.' }
docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) { Fail ('ОШИБКА: образ не найден: ' + $Image) }
New-Item -ItemType Directory -Force -Path $env:OUT | Out-Null
$busy = @(Get-ChildItem -Force -LiteralPath $env:OUT -ErrorAction SilentlyContinue)
if ($busy.Count -gt 0) { Fail ('ОШИБКА: каталог OUT не пуст, журнал не дописывается: ' + $env:OUT) }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
try { docker rm -f $Name 2>$null | Out-Null } catch { }
$publish = @()
if ($View -eq 'web') { $publish = @('-p', ($Port + ':8080')) }
[Console]::Out.WriteLine('Профиль: configured_straight, 2.1 x 3.0 м, s=1..120 м. Это допущение, не восстановленная траектория.')
[Console]::Out.WriteLine("Топик $Topic. Workers $Workers. Картинка enabled=$VizEnabled, предел $VizHz Гц. Просмотр $View.")
docker run -d --name $Name --memory 6g --shm-size 512m --user 1000:1000 -e HOME=/output -e PYTHONUNBUFFERED=1 `
  -e ROS_LOCALHOST_ONLY=1 -e "ROS_DOMAIN_ID=$Domain" -e "VIEW=$View" -e "HOST_PORT=$Port" `
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds.xml `
  @publish `
  -v "${env:OUT}:/output" `
  -v "${env:BAG}:/bag:ro" `
  -v "${Here}\demo_keep.sh:/demo_keep.sh:ro" `
  -v "${Here}\play_inside.sh:/play_inside.sh:ro" `
  -v "${Here}\stop_node.sh:/stop_node.sh:ro" `
  -v "${Here}\fastdds.xml:/fastdds.xml:ro" `
  --entrypoint bash $Image /demo_keep.sh `
  "input_topic:=$Topic" corridor_source:=configured_straight `
  "frame_budget_s:=$Budget" playback_mode:=stream `
  "compute_backend:=$Backend" "workers:=$Workers" queue_policy:=keep_latest subscription_depth:=10 `
  "publish_visualization:=true" "visualization_enabled:=$VizEnabled" "visualization_rate_hz:='$VizHz'" output_dir:=/output
if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: нода не запустилась.' }
[Console]::Out.WriteLine("Нода $Name запущена. Журнал $env:OUT")
[Console]::Out.WriteLine('Плеер запускается отдельно. Закрытие демонстрации: docker stop ' + $Name)
exit 0
