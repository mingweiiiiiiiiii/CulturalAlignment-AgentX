# %%

# jupyter style type script to experiment with visualizing results
!pip install pandas matplotlib seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns


# %%
eval_results_csv = "results/eval_results_20250507_211340.csv"
print("csv_data", csv_data)
print(csv_data.head())
print(csv_data.columns)


# %%
# get average scores based on the type column
# group by type column and get the mean of all numeric columns

average_scores = csv_data.groupby('type').mean(numeric_only=True)
print(average_scores)
# remove the id column
average_scores = average_scores.drop(columns=['id'])


# %%
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
axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]  # Ensure axes is iterable

# Create a bar plot for each metric
for i, metric in enumerate(metrics):
    if i < len(axes):  # Only plot if we have a subplot available
        average_scores[metric].plot(kind='bar', ax=axes[i], color='skyblue')
        axes[i].set_title(f'{metric}')
        axes[i].set_xlabel('Type')
        axes[i].set_ylabel('Value')
        
        # Add value labels on top of each bar
        for j, v in enumerate(average_scores[metric]):
            axes[i].text(j, v + (v * 0.02), f'{v:.2f}', ha='center')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Individual comparison of metrics by type', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



# %%
# Create a single plot with log scale to show all metrics

# Make a copy to avoid modifying original data
log_data = average_scores.copy()

# Handle any zeros or negative values before applying log scale
# Replace zeros with a small value (e.g., 0.01) to avoid log(0) errors
small_value = 0.01
for col in log_data.columns:
    log_data[col] = log_data[col].apply(lambda x: small_value if x <= 0 else x)

# Plot with log scale
plt.figure(figsize=(14, 8))
ax = log_data.plot(kind='bar', logy=True)  # Use log scale on y-axis

# Enhance the plot appearance
plt.title('Comparison of all metrics by type (log scale)', fontsize=16)
plt.xlabel('Type', fontsize=12)
plt.ylabel('Value (log scale)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Add value annotations on the bars
for i, container in enumerate(ax.containers):
    metric_name = log_data.columns[i]
    values = log_data[metric_name].values
    for j, v in enumerate(values):
        # Get original value before log
        original_value = average_scores.iloc[j][metric_name]
        # Position the text slightly above the bar
        ax.text(j + (i-len(ax.containers)/2+0.5)*0.8/len(ax.containers),
                v * 1.1,
                f'{original_value:.2f}',
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=8)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Create a bar chart with metrics on x-axis and types as bar groups
# Transpose the data for better comparison
transposed_data = average_scores.transpose()

# Plot the transposed data
plt.figure(figsize=(14, 8))

# Create the bar chart with metrics on x-axis and color-coded bars for baseline vs model
transposed_data.plot(kind='bar', width=0.7)

# Enhance the plot appearance
plt.title('Comparison of baseline vs model across all metrics', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Type')

# Add value annotations to the bars
for i, container in enumerate(plt.gca().containers):
    plt.gca().bar_label(container, fmt='%.2f', padding=3, rotation=90, fontsize=8)

plt.tight_layout()
plt.show()

# %%
# Create the same chart but with log scale for better visibility of small values
plt.figure(figsize=(14, 8))

# Handle zeros or negative values
log_transposed = transposed_data.copy()
for col in log_transposed.columns:
    log_transposed[col] = log_transposed[col].apply(
        lambda x: 0.01 if x <= 0 else x)

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
        metric_name = transposed_data.index[j]
        value = transposed_data.loc[metric_name, type_name]
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2., height*1.1,
                f'{value:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)

plt.tight_layout()
plt.show()

 # %%
