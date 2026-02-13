#!/bin/bash

# Скрипт для создания conda окружения для MultiKernelBench API

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Параметры по умолчанию
ENV_NAME=${ENV_NAME:-"bench_api"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MultiKernelBench API - Conda Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Проверка наличия conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found${NC}"
    echo "Please install Anaconda or Miniconda first"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Environment name: $ENV_NAME"
echo "  Python version: $PYTHON_VERSION"
echo ""

# Проверка существования окружения
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists${NC}"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Создание окружения
echo -e "${GREEN}Creating conda environment...${NC}"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Активация окружения
echo -e "${GREEN}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Установка зависимостей
echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
if [ -f "requirements.txt" ]; then
    # Использование временной директории для pip cache и установки
    # Создаем временную директорию в текущей папке
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TMP_DIR="$SCRIPT_DIR/.pip_temp_$$"
    
    mkdir -p "$TMP_DIR/pip_cache"
    mkdir -p "$TMP_DIR/build"
    echo -e "${YELLOW}Using temporary directory: $TMP_DIR${NC}"
    
    # Установка с использованием временной директории
    export TMPDIR="$TMP_DIR/build"
    export PIP_CACHE_DIR="$TMP_DIR/pip_cache"
    export TMP="$TMP_DIR/build"
    export TEMP="$TMP_DIR/build"
    
    pip install --cache-dir="$TMP_DIR/pip_cache" -r requirements.txt
    
    # Очистка временной директории после установки
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf "$TMP_DIR"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""

