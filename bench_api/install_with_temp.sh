#!/bin/bash

# Скрипт для установки зависимостей с использованием временной директории
# Используется когда основное дисковое пространство ограничено

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installing dependencies with temp dir${NC}"
echo -e "${GREEN}========================================${NC}"

# Проверка активации conda окружения
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}Warning: Conda environment not activated${NC}"
    echo "Activating bench_api environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bench_api 2>/dev/null || {
        echo -e "${RED}Error: Could not activate bench_api environment${NC}"
        echo "Please create it first with: ./setup_conda.sh"
        exit 1
    }
fi

# Определение временной директории - создаем в папке где запускается скрипт
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/.pip_temp_$$"

mkdir -p "$TEMP_DIR"
echo -e "${YELLOW}Using temporary directory: $TEMP_DIR${NC}"

# Настройка переменных окружения для pip
export TMPDIR="$TEMP_DIR/build"
export PIP_CACHE_DIR="$TEMP_DIR/pip_cache"
export TMP="$TEMP_DIR/build"
export TEMP="$TEMP_DIR/build"

# Создание поддиректорий
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$TEMP_DIR/build"

# Установка зависимостей
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}Installing dependencies...${NC}"
    echo "This may take a while and use temporary space..."
    
    pip install --cache-dir="$PIP_CACHE_DIR" -r requirements.txt
    
    echo -e "${GREEN}Installation completed!${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

# Очистка временных файлов
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"

echo -e "${GREEN}Done!${NC}"

