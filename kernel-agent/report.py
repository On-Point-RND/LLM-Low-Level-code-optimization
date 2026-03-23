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
            categories = sorted(method_df['kernel_category'].unique())

            for cat in categories:
                cat_kernels = current_df[current_df['kernel_category'] == cat]
                # Fix denominator to full kernel set across all iterations
                unique_kernels = method_df[method_df['kernel_category'] == cat]['kernel_name'].unique()
                total = len(unique_kernels)

                comp_count = 0
                pass_count = 0
                su_count = 0
                all_best_speedups = []

                for kernel in unique_kernels:
                    k_df = cat_kernels[cat_kernels['kernel_name'] == kernel]

                    # Cumulative stats
                    if k_df['compiled'].any():
                        comp_count += 1

                    if (k_df['correctness'] == True).any():
                        pass_count += 1

                    if (k_df['correctness'] & (k_df['speedup'] > 1.0)).any():
                        su_count += 1

                    # Best speedup over iterations 0..k;
                    # exclude non-passing kernels and kernels with no speedup measurement
                    correct_at_k = k_df[k_df['correctness'] == True]
                    if not correct_at_k.empty:
                        best = correct_at_k['speedup'].max()
                        if best > 0:
                            all_best_speedups.append(best)

                avg_su = statistics.geometric_mean(all_best_speedups) if all_best_speedups else 1.0
                max_su = max(all_best_speedups) if all_best_speedups else 1.0

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate stats report from experiment results.'
    )
    parser.add_argument('--exp-id', type=str, required=True, help='Experiment ID')
    args = parser.parse_args()

    generate_report(args.exp_id)
