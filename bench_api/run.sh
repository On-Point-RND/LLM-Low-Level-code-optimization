#!/bin/bash

# Скрипт для запуска KernelBench API сервиса

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Параметры по умолчанию
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8123"}
WORKERS=${WORKERS:-"1"}
RELOAD=${RELOAD:-"false"}
LOG_LEVEL=${LOG_LEVEL:-"debug"}
export DEVICE_IDS="${DEVICE_IDS:-0}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}KernelBench API Server${NC}"
echo -e "${GREEN}========================================${NC}"

# Параметры запуска
UVICORN_ARGS="app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL"

if [ "$RELOAD" = "true" ]; then
    UVICORN_ARGS="$UVICORN_ARGS --reload"
    echo -e "${YELLOW}Running in development mode (reload enabled)${NC}"
fi

if [ "$WORKERS" != "1" ] && [ "$RELOAD" != "true" ]; then
    UVICORN_ARGS="$UVICORN_ARGS --workers $WORKERS"
    echo -e "${YELLOW}Running with $WORKERS workers${NC}"
fi

echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Reload: $RELOAD"
echo "  Log Level: $LOG_LEVEL"
echo "  DEVICE_IDS: $DEVICE_IDS"
WORKSPACE_TMP=${WORKSPACE_TMP:-"${TMPDIR:-/tmp}/bench_api_tmp"}
mkdir -p "$WORKSPACE_TMP"
mkdir -p "$WORKSPACE_TMP/cuda_cache"
export CUDA_BUILD_ROOT="$WORKSPACE_TMP"
export CUDA_CACHE_PATH="$WORKSPACE_TMP/cuda_cache"
export TMPDIR="$WORKSPACE_TMP"
export TMP="$WORKSPACE_TMP"
export TEMP="$WORKSPACE_TMP"

echo "Environment:"
echo "  TMPDIR: $TMPDIR"
echo ""
echo "Starting server..."
echo ""

# Запуск сервера
exec python3 -m uvicorn $UVICORN_ARGS
