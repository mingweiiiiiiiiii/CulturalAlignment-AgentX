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

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    csv_data = pd.read_csv(csv_file_path)
    print("Loading data from:", csv_file_path)
    print(csv_data.head())
    print(csv_data.columns)

    # Group by 'type' and compute averages
    average_scores = csv_data.groupby('type').mean(numeric_only=True)
    print(average_scores)

    # Drop non-numeric or irrelevant columns
    for col in ['id', 'timestamp']:
        if col in average_scores.columns:
            average_scores.drop(columns=[col], inplace=True)

    # Set seaborn aesthetics
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    # Individual metric bar charts
    metrics = average_scores.columns.tolist()
    num_metrics = len(metrics)
    n_cols = 3 if num_metrics > 3 else num_metrics
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x=average_scores.index, y=average_scores[metric],
                    hue=average_scores.index, dodge=False,
                    palette="Blues_d", ax=ax, legend=False)
        ax.set_title(metric)
        ax.set_xlabel('Type')
        ax.set_ylabel('Value')

        # Annotate bar values
        for j, val in enumerate(average_scores[metric]):
            ax.text(j, val + (val * 0.02), f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Individual Comparison of Metrics by Type', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(f"viz/individual_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === Log-scale grouped bar chart ===
    transposed_data = average_scores.transpose().reset_index()
    melted_data = pd.melt(transposed_data, id_vars='index')
    melted_data.columns = ['Metric', 'Type', 'Value']
    melted_data['LogValue'] = melted_data['Value'].apply(
        lambda x: 0.01 if pd.isna(x) or x <= 0 else min(x, 1e6)
    )

    plt.figure(figsize=(16, 9))
    ax = sns.barplot(data=melted_data, x='Metric', y='LogValue', hue='Type', palette='Set2')
    ax.set_yscale("log")

    # Annotate each bar with its height (original value)
    for container in ax.containers:
        if not container:
            continue
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8, rotation=90)

    ax.set_title('Comparison of Baseline vs Model Across All Metrics (Log Scale)', fontsize=18, pad=15)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Value (Log Scale)', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
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
