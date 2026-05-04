#!/usr/bin/env python3
"""
Простой пример: CUDA kernel для ReLU.
Отправляет код через API и получает результаты производительности.
"""

import os
import requests
import json
import uuid
import time

BASE_URL = "http://localhost:8123"


def main():
    cuda_code = """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA C++ kernel код
relu_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] > 0) ? x[idx] : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
\"\"\"

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor x);"

# Компиляция CUDA кода
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = relu

    def forward(self, x):
        return self.relu.relu_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
"""

    # Настройки MLflow
    experiment_name = "my_experiments"  # Название папки с экспериментами в MLflow
    run_name = (
        f"relu_cuda_{uuid.uuid4().hex[:8]}_{int(time.time())}"  # Имя конкретного run
    )
    num_trials = 100  # Количество триалов для замеров производительности (можно уменьшить для отладки)

    # Формируем запрос
    payload = {
        "torch_compile": False,
        "language": "cuda",  # Бэкенд: CUDA GPU
        "function": "relu",
        "function_code": cuda_code,  # CUDA C++ kernel код
        "feedback": None,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "num_trials": num_trials,
    }

    print("=" * 70)
    print("Пример: CUDA kernel для ReLU")
    print("=" * 70)
    print(f"\nОтправка запроса:")
    print(f"  Бэкенд: CUDA")
    print(f"  Функция: relu")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Run name: {run_name}")
    print(f"  Num trials: {num_trials}")
    print(f"  Размер кода: {len(cuda_code)} символов")

    # Отправляем запрос
    try:
        response = requests.post(f"{BASE_URL}/evaluate", json=payload, timeout=300)

        print(f"\nОтвет сервера:")
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(data)
            print(f"\n✓ Успешно!")
            print(f"\nРезультаты:")
            print(f"  Run name: {data.get('run_name')}")
            print(f"  Function: {data.get('function')}")
            print(f"  Language: {data.get('language')}")
            print(f"  Hardware: {data.get('hardware')}")
            print(f"  Compiled: {data.get('compiled')}")
            print(f"  Correctness: {data.get('correctness')}")

            if data.get("performance"):
                perf = data["performance"]
                print(f"\nПроизводительность:")
                print(f"  Mean: {perf.get('mean')} ms")
                print(f"  Std: {perf.get('std')} ms")
                print(f"  Min: {perf.get('min')} ms")
                print(f"  Max: {perf.get('max')} ms")
                print(f"  Trials: {perf.get('num_trials')}")

            if data.get("baseline"):
                baseline = data["baseline"]
                print(f"\nBaseline:")
                print(f"  Mean: {baseline.get('mean')} ms")
                print(f"  Std: {baseline.get('std')} ms")
                print(f"  Min: {baseline.get('min')} ms")
                print(f"  Max: {baseline.get('max')} ms")

            if data.get("speedup") is not None:
                print(f"\nУскорение: {data.get('speedup'):.3f}x")

            if data.get("function_code"):
                code_preview = data.get("function_code")
                if len(code_preview) > 200:
                    code_preview = code_preview[:200] + "..."
                print(f"\nКод (первые 200 символов):")
                print(f"  {code_preview}")

            if data.get("baseline_function_code"):
                baseline_code_preview = data.get("baseline_function_code")
                if len(baseline_code_preview) > 200:
                    baseline_code_preview = baseline_code_preview[:200] + "..."
                print(f"\nBaseline код (первые 200 символов):")
                print(f"  {baseline_code_preview}")

            if data.get("compile_info"):
                print(f"\nCompile info: {data.get('compile_info')[:100]}...")

            if data.get("correctness_info"):
                print(f"\nCorrectness info: {data.get('correctness_info')[:100]}...")

            print(f"\nMLflow UI: {os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')}")
            print(f"Experiment: {experiment_name}")
            print(f"Run name: {run_name}")
        else:
            print(f"\n✗ Ошибка:")
            print(f"  {response.text}")

    except Exception as e:
        print(f"\n✗ Исключение: {e}")


if __name__ == "__main__":
    main()
