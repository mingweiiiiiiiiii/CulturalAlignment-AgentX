#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_visualizations(csv_file_path):
    # Create viz directory if it doesn't exist
    os.makedirs("viz", exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the CSV data
    csv_data = pd.read_csv(csv_file_path)
    print("Loading data from:", csv_file_path)
    print(csv_data.head())
    print(csv_data.columns)

    # Get average scores based on the type column
    average_scores = csv_data.groupby('type').mean(numeric_only=True)
    print(average_scores)
    # Remove the id column if it exists
    if 'id' in average_scores.columns:
        average_scores = average_scores.drop(columns=['id'])

    # Create individual bar plots for each metric for better comparison
    metrics = average_scores.columns.tolist()
    num_metrics = len(metrics)

    # Calculate grid dimensions
    if num_metrics <= 3:
        n_rows, n_cols = 1, num_metrics
    else:
        n_cols = 3  # 3 plots per row
        n_rows = (num_metrics + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [
        axes]  # Ensure axes is iterable

    # Create a bar plot for each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):  # Only plot if we have a subplot available
            average_scores[metric].plot(
                kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'{metric}')
            axes[i].set_xlabel('Type')
            axes[i].set_ylabel('Value')

            # Add value labels on top of each bar
            for j, v in enumerate(average_scores[metric]):
                axes[i].text(j, v + (v * 0.02), f'{v:.2f}', ha='center')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Individual comparison of metrics by type',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    # Save instead of showing
    plt.savefig(
        f"viz/individual_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create a bar chart with metrics on x-axis and types as bar groups
    # Transpose the data for better comparison
    transposed_data = average_scores.transpose()

    # Create the same chart but with log scale for better visibility of small values
    plt.figure(figsize=(14, 8))

    # Handle zeros, negative, NaN, and extremely large values
    log_transposed = transposed_data.copy()
    for col in log_transposed.columns:
        log_transposed[col] = log_transposed[col].apply(
            lambda x: 0.01 if x <= 0 or pd.isna(x) else min(x, 1e6))  # Cap at 1 million

    # Create the log-scale bar chart
    ax = log_transposed.plot(kind='bar', width=0.7, logy=True)

    # Enhance the plot appearance
    plt.title(
        'Comparison of baseline vs model across all metrics (log scale)', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Value (log scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(title='Type')

    # Add value annotations to the bars (original values)
    for i, container in enumerate(plt.gca().containers):
        type_name = transposed_data.columns[i]
        for j, patch in enumerate(container):
            if j < len(transposed_data.index):  # Ensure index is in range
                metric_name = transposed_data.index[j]
                value = transposed_data.loc[metric_name, type_name]
                if not pd.isna(value):  # Only add text for non-NaN values
                    height = patch.get_height()
                    ax.text(patch.get_x() + patch.get_width()/2., height*1.1,
                           f'{value:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)

    plt.tight_layout()
    # Save instead of showing
    plt.savefig(
        f"viz/comparison_metrics_log_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"All visualizations saved to the 'viz' directory with timestamp {timestamp}")


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate visualizations from evaluation results CSV')
    parser.add_argument('--csv', dest='csv_file', type=str,
                        default="results/eval_results_20250507_211340.csv",
                        help='Path to the CSV file with evaluation results')

    args = parser.parse_args()

    create_visualizations(args.csv_file)


if __name__ == "__main__":
    main()
