#!/usr/bin/env python3
"""
Compare benchmark results from two JSON files and plot comparison graph
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_times(data1, data2):
    """Extract timing data from both JSON files"""
    # data1 is the cuda results (dict with model names as keys)
    # data2 is the summary results (list of dicts)
    
    # Create mapping from summary.json
    summary_map = {}
    for item in data2:
        model_name = item.get('name', '')
        if 'latency' in item and 'after_tir' in item['latency']:
            mean_ms = item['latency']['after_tir'].get('mean_ms', 0)
            summary_map[model_name] = mean_ms
    
    # Extract common models
    common_models = []
    cuda_times = []
    summary_times = []
    
    for model_name, cuda_data in data1.items():
        if model_name in summary_map:
            cuda_mean = cuda_data.get('mean', 0)  # in milliseconds
            summary_mean = summary_map[model_name]  # in milliseconds
            
            common_models.append(model_name)
            cuda_times.append(cuda_mean)
            summary_times.append(summary_mean)
    
    return common_models, cuda_times, summary_times

def plot_comparison(models, cuda_times, summary_times, output_file='comparison.png'):
    """Plot comparison graph"""
    # Sort by model name for better readability
    sorted_data = sorted(zip(models, cuda_times, summary_times), key=lambda x: x[0])
    models_sorted, cuda_sorted, summary_sorted = zip(*sorted_data)
    
    x = np.arange(len(models_sorted))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create bars
    bars1 = ax.bar(x - width/2, cuda_sorted, width, label='CUDA', alpha=0.8)
    bars2 = ax.bar(x + width/2, summary_sorted, width, label='TVM', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Benchmark Comparison: CUDA vs TVM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=6, rotation=90)
    
    # Only label if there are not too many models
    if len(models_sorted) <= 20:
        autolabel(bars1)
        autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {output_file}")
    
    # Also create a log-scale version for better visibility
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    bars3 = ax2.bar(x - width/2, cuda_sorted, width, label='CUDA', alpha=0.8)
    bars4 = ax2.bar(x + width/2, summary_sorted, width, label='TVM', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Time (ms, log scale)', fontsize=12)
    ax2.set_title('Benchmark Comparison: CUDA vs TVM (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=8)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_log.png'), dpi=150, bbox_inches='tight')
    print(f"Log-scale graph saved to {output_file.replace('.png', '_log.png')}")

def print_statistics(models, cuda_times, summary_times):
    """Print comparison statistics"""
    print("\n" + "="*80)
    print("COMPARISON STATISTICS")
    print("="*80)
    
    ratios = [s/c if c > 0 else 0 for c, s in zip(cuda_times, summary_times)]
    
    print(f"\nTotal models compared: {len(models)}")
    print(f"\nCUDA results:")
    print(f"  Mean time: {np.mean(cuda_times):.4f} ms")
    print(f"  Median time: {np.median(cuda_times):.4f} ms")
    print(f"  Min time: {np.min(cuda_times):.4f} ms")
    print(f"  Max time: {np.max(cuda_times):.4f} ms")
    
    print(f"\nTVM results (summary.json):")
    print(f"  Mean time: {np.mean(summary_times):.4f} ms")
    print(f"  Median time: {np.median(summary_times):.4f} ms")
    print(f"  Min time: {np.min(summary_times):.4f} ms")
    print(f"  Max time: {np.max(summary_times):.4f} ms")
    
    print(f"\nSpeedup ratio (TVM/CUDA):")
    print(f"  Mean ratio: {np.mean(ratios):.4f}x")
    print(f"  Median ratio: {np.median(ratios):.4f}x")
    print(f"  Min ratio: {np.min(ratios):.4f}x")
    print(f"  Max ratio: {np.max(ratios):.4f}x")
    
    # Find models with biggest differences
    diff_abs = [abs(s - c) for c, s in zip(cuda_times, summary_times)]
    diff_rel = [abs(s - c) / c * 100 if c > 0 else 0 for c, s in zip(cuda_times, summary_times)]
    
    print(f"\nTop 10 models with biggest absolute difference:")
    top_abs = sorted(zip(models, diff_abs), key=lambda x: x[1], reverse=True)[:10]
    for model, diff in top_abs:
        idx = models.index(model)
        print(f"  {model}: {diff:.4f} ms (CUDA: {cuda_times[idx]:.4f} ms, TVM: {summary_times[idx]:.4f} ms)")
    
    print(f"\nTop 10 models with biggest relative difference:")
    top_rel = sorted(zip(models, diff_rel), key=lambda x: x[1], reverse=True)[:10]
    for model, diff in top_rel:
        idx = models.index(model)
        print(f"  {model}: {diff:.2f}% (CUDA: {cuda_times[idx]:.4f} ms, TVM: {summary_times[idx]:.4f} ms)")

if __name__ == "__main__":
    import sys
    
    # Default paths (relative to bench directory)
    cuda_file = "res/cuda_NVIDIA A100-SXM4-80GB.json"
    summary_file = "res/summary.json"
    output_file = "res/comparison.png"
    
    if len(sys.argv) > 1:
        cuda_file = sys.argv[1]
    if len(sys.argv) > 2:
        summary_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Load data
    print(f"Loading {cuda_file}...")
    cuda_data = load_json(cuda_file)
    
    print(f"Loading {summary_file}...")
    summary_data = load_json(summary_file)
    
    # Extract times
    models, cuda_times, summary_times = extract_times(cuda_data, summary_data)
    
    print(f"\nFound {len(models)} common models")
    
    # Print statistics
    print_statistics(models, cuda_times, summary_times)
    
    # Plot comparison
    print(f"\nGenerating comparison graph...")
    plot_comparison(models, cuda_times, summary_times, output_file)
    
    print("\nDone!")
