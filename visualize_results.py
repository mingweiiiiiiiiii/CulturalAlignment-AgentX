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

    # Set Seaborn styling
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Create individual vertical bar plots for each metric
    metrics = average_scores.columns.tolist()
    num_metrics = len(metrics)

    # Calculate subplot grid dimensions
    n_cols = 3 if num_metrics > 3 else num_metrics
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x=average_scores.index, y=average_scores[metric],
                    palette="Blues_d", ax=ax)
        ax.set_title(f'{metric}')
        ax.set_xlabel('Type')
        ax.set_ylabel('Value')

        # Annotate bar values
        for j, val in enumerate(average_scores[metric]):
            ax.text(j, val + (val * 0.02), f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Individual Comparison of Metrics by Type', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(f"viz/individual_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Transpose data for grouped metric comparison
    transposed_data = average_scores.transpose()

    # Melt into long format
    melted_data = transposed_data.reset_index().melt(id_vars='index')
    melted_data.columns = ['Metric', 'Type', 'Value']

    # Replace invalid values for log scale
    melted_data['LogValue'] = melted_data['Value'].apply(
        lambda x: 0.01 if pd.isna(x) or x <= 0 else min(x, 1e6)
    )

    # Grouped bar chart with log scale
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(data=melted_data, x='Metric', y='LogValue', hue='Type', log=True, palette='Set2')

    # Annotate bars with original (non-log) values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, rotation=90, padding=2)

    ax.set_title('Comparison of Baseline vs Model Across All Metrics (Log Scale)', fontsize=18, pad=15)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Value (Log Scale)', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Improve legend placement
    plt.legend(title='Type', title_fontsize=12, fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(f"viz/comparison_metrics_log_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All visualizations saved to the 'viz' directory with timestamp {timestamp}")
    
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
