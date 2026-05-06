# TVM LLM benchmark

Единый запускной пакет: Relax → LLM-TIR (или MetaSchedule) → профилирование по задачам из KernelBench.

## Запуск

```bash
cd /path/to/ready
python run_benchmark.py --gpu 0
```

Опционально другой конфиг:

```bash
python run_benchmark.py --gpu 0 --config /path/to/config.yaml
```

Одна задача (внутренний режим; обычно вызывается оркестратором):

```bash
python run_benchmark.py --run-single /path/to/model.py --gpu 0 --config config.yaml
```

## Зависимости

- Python-пакеты: как минимум **PyYAML**, **torch**, **numpy**, **pydantic** (см. импорты в `core/`).
- **Apache TVM** с Python bindings: положите сборку в `ready/tvm/python` или поправьте `sys.path` в начале `run_benchmark.py`.
- CUDA и модели на стороне эндпоинта LLM — по вашей среде.

## Конфиг (`config.yaml`)

| Ключ | Назначение |
|------|------------|
| `kernelbench_root` | Корень с baseline `.py` (абсолютный путь или относительный к каталогу `run_benchmark.py`). |
| `transform_tir_mode` | `llm_transform_tir` или режим с MetaSchedule (`tir_transforms` и т.д. — см. `core/pipeline.py`). |
| `llm_tir_strategy` | При `llm_transform_tir`: `feedback` (многошаговый цикл, шаги — `llm_tir_max_steps`) или `sampling` (5 сэмплов + финал, кэш в `res/.../llm_logs_tir`). |
| `agents.*` | URL, ключ, модель для OpenAI-совместимого API. |
| `gpu_id` | Значение по умолчанию в YAML; фактическое устройство задаётся **`--gpu`** у CLI. |

Результаты пишутся в **`res/`** рядом с раннером (артефакты по путям моделей относительно `kernelbench_root`).

## Эталонные латентности (опционально)

Для порогов профилирования можно положить `baseline/baseline_time_torch_compile.json` рядом с родителем каталога `res/` (см. `_load_baseline_mean_ms` в `core/pipeline.py`).
