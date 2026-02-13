# MultiKernelBench API

FastAPI сервис для оценки производительности кернелов с использованием MultiKernelBench.

## Установка

### Вариант 1: Использование Conda (рекомендуется)

#### Автоматическая установка через скрипт:

```bash
./setup_conda.sh
```

Скрипт создаст окружение `bench_api` с Python 3.10 и установит все зависимости.

#### Ручная установка через environment.yml:

```bash
conda env create -f environment.yml

# Активация окружения
conda activate bench_api

pip install torch==2.1.0

# Для CUDA:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Для NPU (Ascend):
pip install torch-npu==2.1.0.post12
```

#### Ручное создание окружения:

```bash
# Создание окружения
conda create -n bench_api python=3.10 -y

# Активация окружения
conda activate bench_api

# Установка зависимостей
pip install -r requirements.txt

# Установка PyTorch (см. варианты выше)
```

#### Если не хватает места на диске (использование временной директории):

Если при установке возникает ошибка "No space left on device", используйте скрипт с временной директорией:

```bash
# Создание окружения (если еще не создано)
conda create -n bench_api python=3.10 -y
conda activate bench_api

# Установка с использованием временной директории
./install_with_temp.sh
```

Или вручную с указанием временной директории:

```bash
conda activate bench_api

# Установка с использованием временной директории
export TMPDIR=/tmp/pip_install_$$
export PIP_CACHE_DIR="$TMPDIR/pip_cache"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"

pip install --cache-dir="$PIP_CACHE_DIR" --build="$TMPDIR/build" -r requirements.txt

# Очистка после установки
rm -rf "$TMPDIR"
```

### Вариант 2: Использование pip напрямую

```bash
pip install -r requirements.txt
```

**Примечание:** Для работы с GPU/NPU может потребоваться установка специфичных версий PyTorch.

## Запуск

### Вариант 1: Использование bash скрипта (рекомендуется)

```bash
./run.sh
```

С параметрами через переменные окружения:

```bash
HOST=0.0.0.0 PORT=8000 WORKERS=1 RELOAD=true LOG_LEVEL=info ./run.sh
```

### Вариант 2: Использование Python скрипта

```bash
python3 start_server.py
```

С параметрами:

```bash
python3 start_server.py --host 0.0.0.0 --port 8000 --workers 1 --reload --log-level info
```

### Вариант 3: Прямой запуск через uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Или с reload для разработки:

```bash
uvicorn app.main:app --reload
```

## Тестирование

Для проверки работы API используйте тестовые скрипты:

```bash
# Быстрые тесты (только help и root endpoints)
python3 test_api.py

# Комплексные тесты (включая baseline и evaluate - требует GPU)
python3 test_api_comprehensive.py

# Расширенные тесты (больше сценариев, разные функции, граничные случаи)
python3 test_api_extended.py

# Полный набор тестов с опциями
python3 test_api_comprehensive.py --full  # Включает долгие тесты
```

Перед запуском тестов убедитесь, что сервер запущен:

```bash
# В одном терминале
./run.sh

# В другом терминале
python3 test_api_comprehensive.py --full
# или
python3 test_api_extended.py
```

## API Endpoints

### GET /help

Получить информацию о доступных языках, функциях и категориях.

**Ответ:**
```json
{
  "languages": ["cuda", "triton", ...],
  "functions": {
    "leaky_relu": "activation",
    ...
  },
  "categories": ["activation", "matmul", ...],
  "endpoints": {...}
}
```

### POST /baseline

Получить бейзлайн производительности для функции или всех функций языка.

**Запрос:**
```json
{
  "language": "cuda",
  "function": "leaky_relu"  // или "all" для всех функций
}
```

**Ответ для одной функции:**
```json
{
  "function": "leaky_relu",
  "language": "cuda",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "baseline": {
    "mean": 0.123,
    "std": 0.001,
    "min": 0.120,
    "max": 0.125,
    "num_trials": 100
  },
  "cached": true
}
```

**Ответ для всех функций (function="all"):**
```json
{
  "function": "all",
  "language": "cuda",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "baselines": {
    "leaky_relu": {...},
    "relu": {...},
    ...
  },
  "cached": true
}
```

### POST /evaluate

Оценить производительность кернела.

**Запрос:**
```json
{
  "torch_compile": false,
  "language": "cuda",
  "function": "leaky_relu",
  "function_code": "...",
  "feedback": "optional feedback",
  "experiment_name": "my_experiments",
  "run_name": "leaky_relu_run_001"
}
```

**Примечание:** 
- `experiment_name` - опциональное поле, название папки с экспериментами в MLflow. Если не указано, используется значение из `MLFLOW_EXPERIMENT_NAME`.
- `run_name` - опциональное поле, имя конкретного run внутри эксперимента. Если указано, будет возвращено в ответе.

**Ответ:**
```json
{
  "function": "leaky_relu",
  "language": "cuda",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "compiled": true,
  "correctness": true,
  "performance": {
    "mean": 0.115,
    "std": 0.001,
    "min": 0.113,
    "max": 0.118,
    "num_trials": 100
  },
  "compile_info": null,
  "correctness_info": null,
  "run_name": "leaky_relu_run_001"
}
```

## Конфигурация

Настройки можно задать через переменные окружения:

### MLflow (логирование экспериментов)

- `MLFLOW_TRACKING_URI` - URI MLflow сервера (по умолчанию: `http://195.209.214.105:5050`)
- `MLFLOW_EXPERIMENT_NAME` - Имя эксперимента в MLflow по умолчанию (по умолчанию: `MultiKernelBench`, используется если `experiment_name` не указан в запросе)
- `ENABLE_MLFLOW` - Включить/выключить логирование в MLflow (по умолчанию: `true`)

При каждом вызове `/evaluate` с `experiment_name` или `run_name` результаты автоматически логируются в MLflow:
- **Эксперимент**: создается/используется эксперимент с именем из `experiment_name` (или `MLFLOW_EXPERIMENT_NAME` если не указано)
- **Параметры**: function, language, hardware, torch_compile, compiled
- **Метрики**: compiled (0/1), correctness (0/1), performance_mean, performance_std, performance_min, performance_max, performance_num_trials
- **Теги**: run_name, experiment_name (если указаны)
- **Артефакты**: compile_info.txt, correctness_info.txt, feedback.txt (если длинный)

Пример использования:
```bash
# Включить MLflow (по умолчанию включен)
ENABLE_MLFLOW=true ./run.sh

# Изменить URI MLflow сервера
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 ./run.sh

# Отключить MLflow
ENABLE_MLFLOW=false ./run.sh
```

### Другие настройки

- `MULTIKERNELBENCH_PATH` - путь к MultiKernelBench (по умолчанию `./MultiKernelBench`)
- `BASELINES_DIR` - путь к директории с бейзлайнами
- `REFERENCE_DIR` - путь к reference реализациям
- `LOG_LEVEL` - уровень логирования (INFO, DEBUG, etc.)
- `ENABLE_TORCH_COMPILE` - глобальный флаг для torch.compile

## Структура проекта

```
bench_api/
├── app/
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Настройки
│   ├── models.py            # Pydantic модели
│   ├── services/            # Бизнес-логика
│   └── core/                # Обертка над MultiKernelBench
├── MultiKernelBench/         # MultiKernelBench
├── run.sh                   # Bash скрипт для запуска
├── start_server.py          # Python скрипт для запуска
├── test_api.py              # Тестовый скрипт
├── setup_conda.sh           # Скрипт для создания conda окружения
├── install_with_temp.sh      # Скрипт для установки с использованием временной директории
├── environment.yml          # Conda environment файл
├── requirements.txt         # Python зависимости
└── README.md
```

## Примеры использования

### Получение информации о доступных функциях

```bash
curl http://localhost:8000/help
```

### Получение бейзлайна для одной функции

```bash
curl -X POST http://localhost:8000/baseline \
  -H "Content-Type: application/json" \
  -d '{"language": "cuda", "function": "leaky_relu"}'
```

### Получение всех бейзлайнов для языка

```bash
curl -X POST http://localhost:8000/baseline \
  -H "Content-Type: application/json" \
  -d '{"language": "cuda", "function": "all"}'
```

### Оценка производительности кернела

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "torch_compile": false,
    "language": "cuda",
    "function": "leaky_relu",
    "function_code": "import torch\nimport torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n    def forward(self, x):\n        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)"
  }'
```

