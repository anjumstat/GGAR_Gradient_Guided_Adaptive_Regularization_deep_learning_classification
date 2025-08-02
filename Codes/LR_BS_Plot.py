# -*- coding: utf-8 -*-
"""
Created on [Date]
Updated with consistent y-axis limits (0.99 to 1.001)
@author: H.A.R
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read CSV
data = pd.read_csv(r'E:\GGAR\Combined_CSVs\Average_Metrics.csv')

# Parameters
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128, 256]
methods = ['GGAR', 'Adaptive', 'FixedL1', 'FixedL2', 'ElasticNet', 'MLP']
colors = ['b', 'g', 'r', 'c', 'm', 'y']
line_styles = [':', '--', '-.', ':', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', '*']

# Create subplots (3 rows × 3 columns) with adjusted figure size
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 16), sharey=True)
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# Adjust global plot parameters
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['lines.markersize'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

label_font = {'fontsize': 14}
legend_font = {'title_fontsize': 14, 'fontsize': 12}
titles = ['Learning Rate: 0.01', 'Learning Rate: 0.001', 'Learning Rate: 0.0001']

# Define the datasets we want to plot (rows)
datasets = ['Train', 'Validation', 'Test']

for row_idx, dataset in enumerate(datasets):
    dataset_data = data[data['Dataset'] == dataset]
    
    for col_idx, lr in enumerate(learning_rates):
        ax = axes[row_idx][col_idx]
        lr_data = dataset_data[dataset_data['learning_rate'] == lr]

        for method, color, ls, marker in zip(methods, colors, line_styles, markers):
            method_data = lr_data[lr_data['method'] == method].sort_values('batch_size')
            if not method_data.empty:
                # Add small jitter to prevent overlapping points
                jitter = np.random.uniform(-0.5, 0.5, len(method_data))
                ax.plot(
                    method_data['batch_size'] + jitter,
                    method_data['Accuracy'],
                    linestyle=ls,
                    color=color,
                    marker=marker,
                    label=method,
                    alpha=0.7,
                    linewidth=2
                )

        ax.set_title(titles[col_idx], fontsize=14, pad=15)
        ax.set_xlabel('Batch Size', fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f'{dataset} Accuracy', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_xticks(batch_sizes)
        ax.set_ylim(0.99, 1.001)  # Changed to match your first version's y-axis limits
        ax.grid(True, which="both", ls="--", alpha=0.2)

        if row_idx == 0 and col_idx == 2:
            ax.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left', **legend_font)

# Adjust main title position and padding
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle('Average Accuracies (Test, Validation, Training) vs Batch Size at Different Learning Rates', 
             fontsize=16, y=0.98)

# Create output directory if it doesn't exist
output_dir = r'E:\GGAR\GGAR_combined'
os.makedirs(output_dir, exist_ok=True)

# Save both formats
base_filename = os.path.join(output_dir, 'all_accuracies_new_format')

# Save as PDF (vector format)
pdf_path = f"{base_filename}.pdf"
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✅ PDF plot saved at: {pdf_path}")

# Save as PNG (raster format)
png_path = f"{base_filename}.png"
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
print(f"✅ PNG plot saved at: {png_path}")

plt.show()