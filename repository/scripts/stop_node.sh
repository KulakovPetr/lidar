#!/bin/bash
# Stop only the detector so it can flush the journal. The viewer process stays.
set +u
pid=$(ps -eo pid,args | awk '/obstacle_node/ && !/awk/ {print $1; exit}')
echo "NODE_PID=${pid:-}"
if [ -n "${pid:-}" ]; then
  kill -INT "$pid"
  echo "SIGINT_SENT"
else
  echo "NODE_NOT_FOUND"
  exit 1
fi
