# Запись демонстрации: нода, проверка подписки, просмотр, затем ros2 bag play.
# UTF-8 с BOM для Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
function Fail([string]$Message) { [Console]::Out.WriteLine($Message); exit 1 }
function Read-Shared([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return '' }
  $stream = New-Object IO.FileStream($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {
    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
    return $reader.ReadToEnd()
  } finally {
    $reader.Dispose()
    $stream.Dispose()
  }
}
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$settings = Join-Path $Here 'demo_settings.ps1'
if (Test-Path -LiteralPath $settings) { . $settings }
if (-not $env:BAG) { Fail 'ОШИБКА: задайте BAG, каталог rosbag2 с metadata.yaml.' }
if (-not $env:OUT) { Fail 'ОШИБКА: задайте пустой каталог OUT.' }
if (-not (Test-Path -LiteralPath $env:BAG)) { Fail ('ОШИБКА: каталог BAG не найден: ' + $env:BAG) }
if (-not (Test-Path -LiteralPath (Join-Path $env:BAG 'metadata.yaml'))) { Fail 'ОШИБКА: в BAG нет metadata.yaml.' }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Name = if ($env:NODE_NAME) { $env:NODE_NAME } else { 'obstacle-node' }
$env:NODE_NAME = $Name
$View = if ($env:VIEW) { $env:VIEW } else { 'web' }
$env:VIEW = $View
$Port = if ($env:PORT) { $env:PORT } else { '8091' }
$env:PORT = $Port
if (-not $env:VIZ_HZ) { $env:VIZ_HZ = '10' }
if (-not $env:RATE) { $env:RATE = '1.0' }
if (-not $env:READ_AHEAD) { $env:READ_AHEAD = '2' }
if (-not $env:WORKERS) { $env:WORKERS = '2' }
if (-not $env:TOPIC) { $env:TOPIC = '/lidar_points' }
$completed = $false
try {
  & (Join-Path $Here 'run_node.ps1')
  if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: run_node.ps1 не запустил ноду.' }
  $ready = $false
  for ($i = 0; $i -lt 40; $i++) {
    $errPath = Join-Path $env:OUT 'node_stderr.txt'
    $outPath = Join-Path $env:OUT 'node_stdout.txt'
    $text = ''
    $text += (Read-Shared $errPath)
    $text += (Read-Shared $outPath)
    if ($text -match 'subscription_ready') { $ready = $true; break }
    if ($text -match 'каталог результата уже занят' -or $text -match 'visualization_rate_hz должен' -or $text -match 'Traceback' -or $text -match 'InvalidParameterTypeException') {
      Fail $text
    }
    Start-Sleep -Seconds 1
  }
  if (-not $ready) { Fail 'ОШИБКА: подписка не стала готова за 40 с. Смотрите node_stderr.txt.' }
  [Console]::Out.WriteLine('Подписка готова. Источник картинки — живые сообщения ROS, не файл.')
  if ($View -eq 'web') {
    $url = 'http://127.0.0.1:' + $Port + '/'
    [Console]::Out.WriteLine('Откройте ' + $url)
    try { Start-Process $url } catch { [Console]::Out.WriteLine('Браузер не открылся. Адрес: ' + $url) }
  } elseif ($View -eq 'rviz') {
    Fail 'ОШИБКА: RViz на этой Windows-команде не запускается. Для RViz используйте run_demo.sh на Ubuntu. Для записи здесь задайте VIEW=web.'
  }
  if ($env:DEMO_NO_PAUSE -ne '1') {
    [Console]::Out.WriteLine('Включите запись экрана. Затем нажмите Enter, и начнётся ros2 bag play.')
    [void](Read-Host 'Enter')
  }
  [Console]::Out.WriteLine('Скорость плеера RATE=' + $env:RATE + '. Это не частота картинки и не обещание 10 Гц.')
  [Console]::Out.WriteLine('Предел новых кадров визуализации: ' + $env:VIZ_HZ + ' Гц.')
  & (Join-Path $Here 'run_play.ps1')
  if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: run_play.ps1 завершился с ошибкой.' }
  $pidFile = Join-Path $env:OUT 'launch.pid'
  $launchPid = ''
  for ($i = 0; $i -lt 10; $i++) {
    if (Test-Path -LiteralPath $pidFile) { $launchPid = (Read-Shared $pidFile).Trim(); break }
    Start-Sleep -Seconds 1
  }
  if (-not $launchPid) { Fail 'ОШИБКА: нет launch.pid, ноду нельзя остановить адресно.' }
  docker exec -u 1000:1000 $Name bash /stop_node.sh
  if ($LASTEXITCODE -ne 0) { Fail 'ОШИБКА: не удалось остановить ноду сигналом SIGINT.' }
  $countsPath = Join-Path $env:OUT 'node_counts.json'
  $flushed = $false
  for ($i = 0; $i -lt 60; $i++) {
    if (Test-Path -LiteralPath $countsPath) {
      $raw = Read-Shared $countsPath
      if ($raw -match '"journal_closed": true') { $flushed = $true; break }
    }
    Start-Sleep -Seconds 1
  }
  if (-not $flushed) { Fail 'ОШИБКА: журнал не подтвердил закрытие. node_counts.json не готов.' }
  [Console]::Out.WriteLine('Счётчики: ' + $countsPath)
  [Console]::Out.WriteLine('Страница остаётся открытой, пока контейнер жив. Закрытие: docker stop ' + $Name)
  $completed = $true
} finally {
  if (-not $completed) {
    try { docker stop $Name | Out-Null } catch { }
  }
}
exit 0
