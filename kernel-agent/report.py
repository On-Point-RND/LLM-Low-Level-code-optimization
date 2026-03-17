import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd


def print_table(title, headers, rows):
    # Define column widths
    col_widths = [20, 10, 10, 10, 10, 10, 10]

    # Format row string
    def format_row(cols):
        return ''.join(f'{str(c):<{w}}' for c, w in zip(cols, col_widths))

    print(f'\n{title}')
    print('-' * sum(col_widths))
    print(format_row(headers))
    print('-' * sum(col_widths))

    for row in rows:
        print(format_row(row))

    print('-' * sum(col_widths))


def generate_report(exp_id):
    file_path = Path(f'data/{exp_id}/results.parquet')
    if not file_path.exists():
        print(f'Error: File not found at {file_path}')
        sys.exit(1)

    df = pd.read_parquet(file_path)

    # Ensure necessary columns exist
    required_cols = {
        'search_method',
        'iteration',
        'kernel_category',
        'kernel_name',
        'model_name',
        'compiled',
        'correctness',
        'speedup',
    }
    if not required_cols.issubset(df.columns):
        print(f'Error: Missing columns in parquet file. Found: {df.columns}')
        sys.exit(1)

    # Convert columns to appropriate types
    df['iteration'] = df['iteration'].astype(int)
    df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce').fillna(0)

    # Get unique search methods
    methods = df['search_method'].unique()

    for method in methods:
        method_df = df[df['search_method'] == method]
        
        # Assume iteration is 0-indexed
        unique_iterations = sorted(method_df['iteration'].unique())

        for iter_idx in unique_iterations:
            k = iter_idx + 1
            print(f'\n\n=== Method: {method} | k = {k} (Iterations 0-{iter_idx}) ===')

            # Filter data up to current iteration (inclusive)
            current_df = method_df[method_df['iteration'] <= iter_idx]

            # --- Aggregation Logic ---
            # Group by Category
            cat_stats = []
            categories = sorted(current_df['kernel_category'].unique())

            for cat in categories:
                cat_kernels = current_df[current_df['kernel_category'] == cat]
                unique_kernels = cat_kernels['kernel_name'].unique()
                total = len(unique_kernels)

                comp_count = 0
                pass_count = 0
                su_count = 0
                passing_at_k_speedups = []

                for kernel in unique_kernels:
                    k_df = cat_kernels[cat_kernels['kernel_name'] == kernel]
                    
                    # Cumulative stats
                    if k_df['compiled'].any():
                        comp_count += 1
                    
                    if (k_df['correctness'] == True).any():
                        pass_count += 1
                    
                    if (k_df['correctness'] & (k_df['speedup'] > 1.0)).any():
                        su_count += 1

                    # Speedup for SPECIFIC iteration k
                    iter_k_df = k_df[k_df['iteration'] == iter_idx]
                    correct_at_k = iter_k_df[iter_k_df['correctness'] == True]
                    if not correct_at_k.empty:
                        passing_at_k_speedups.append(correct_at_k['speedup'].max())

                # Only include positive speedups for geometric mean
                positive_speedups = [s for s in passing_at_k_speedups if s > 0]
                avg_su = statistics.geometric_mean(positive_speedups) if positive_speedups else 0.0
                max_su = max(passing_at_k_speedups) if passing_at_k_speedups else 0.0

                cat_stats.append(
                    [
                        cat,
                        total,
                        f'{comp_count / total * 100:.2f}%',
                        f'{pass_count / total * 100:.2f}%',
                        f'{su_count / total * 100:.2f}%',
                        f'{avg_su:.2f}x',
                        f'{max_su:.2f}x',
                    ]
                )

            print_table(
                'Сводная статистика по категориям:',
                ['Category', 'Total', f'Comp@{k}', f'Pass@{k}', f'SU1@{k}', f'AvgSU@{k}', f'MaxSU@{k}'],
                cat_stats,
            )

            # Group by Model
            model_stats = []
            models = sorted(current_df['model_name'].unique())

            for model in models:
                model_kernels = current_df[current_df['model_name'] == model]
                unique_kernels = model_kernels['kernel_name'].unique()
                total = len(unique_kernels)

                comp_count = 0
                pass_count = 0
                su_count = 0
                passing_at_k_speedups = []

                for kernel in unique_kernels:
                    k_df = model_kernels[model_kernels['kernel_name'] == kernel]
                    
                    # Cumulative stats
                    if k_df['compiled'].any():
                        comp_count += 1
                    
                    if (k_df['correctness'] == True).any():
                        pass_count += 1
                    
                    if (k_df['correctness'] & (k_df['speedup'] > 1.0)).any():
                        su_count += 1

                    # Speedup for SPECIFIC iteration k
                    iter_k_df = k_df[k_df['iteration'] == iter_idx]
                    correct_at_k = iter_k_df[iter_k_df['correctness'] == True]
                    if not correct_at_k.empty:
                        passing_at_k_speedups.append(correct_at_k['speedup'].max())

                # Only include positive speedups for geometric mean
                positive_speedups = [s for s in passing_at_k_speedups if s > 0]
                avg_su = statistics.geometric_mean(positive_speedups) if positive_speedups else 0.0
                max_su = max(passing_at_k_speedups) if passing_at_k_speedups else 0.0

                # Shorten model name for display if too long
                display_name = (model[:15] + '..') if len(model) > 17 else model

                model_stats.append(
                    [
                        display_name,
                        total,
                        f'{comp_count / total * 100:.2f}%',
                        f'{pass_count / total * 100:.2f}%',
                        f'{su_count / total * 100:.2f}%',
                        f'{avg_su:.2f}x',
                        f'{max_su:.2f}x',
                    ]
                )

            print_table(
                'Сводная статистика по моделям:',
                ['Model', 'Total', f'Comp@{k}', f'Pass@{k}', f'SU1@{k}', f'AvgSU@{k}', f'MaxSU@{k}'],
                model_stats,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate stats report from experiment results.'
    )
    parser.add_argument('--exp-id', type=str, required=True, help='Experiment ID')
    args = parser.parse_args()

    generate_report(args.exp_id)
