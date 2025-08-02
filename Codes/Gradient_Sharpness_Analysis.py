

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 15:41:56 2025

@author: H.A.R
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.lines import Line2D

# Configuration
base_dir = Path(r"E:\GGAR\GGAR_combined")
output_dir = base_dir / "consolidated_analysis"
os.makedirs(output_dir, exist_ok=True)

learning_rates = ['0_01', '0_001', '0_0001']
batch_sizes = ['bs_32', 'bs_64', 'bs_128', 'bs_256']
model_names = ['Adaptive', 'ElasticNet', 'FixedL1', 'FixedL2', 'GGAR', 'MLP']

# Define a consistent color palette for models
model_colors = {
    'Adaptive': '#1f77b4',
    'ElasticNet': '#ff7f0e',
    'FixedL1': '#2ca02c',
    'FixedL2': '#d62728',
    'GGAR': '#9467bd',
    'MLP': '#8c564b'
}

# Define line styles for batch sizes
batch_linestyles = {
    'bs_32': '-',
    'bs_64': '--',
    'bs_128': ':',
    'bs_256': '-.'
}

# Data containers
gradient_data = {lr: {model: {bs: [] for bs in batch_sizes} for model in model_names} for lr in learning_rates}
sharpness_data = {lr: {model: {bs: [] for bs in batch_sizes} for model in model_names} for lr in learning_rates}

# Process all learning rates and batch sizes
for lr in learning_rates:
    for bs in batch_sizes:
        exp_dir = base_dir / f"lr_{lr}_{bs}"
        if not exp_dir.exists():
            print(f"⚠️ Missing directory: {exp_dir}")
            continue
            
        print(f"\nProcessing lr: {lr}, batch size: {bs}")
        
        for model in model_names:
            model_dir = exp_dir / f"results_{model}"
            if not model_dir.exists():
                print(f"⚠️ Missing model: {model_dir.name}")
                continue
                
            try:
                # Load all folds
                fold_files = sorted(model_dir.glob("fold*_loss.npy"))
                if not fold_files:
                    print(f"⚠️ No fold files in {model_dir.name}")
                    continue
                    
                # Process each fold
                for fold_file in fold_files:
                    data = np.load(fold_file, allow_pickle=True)
                    
                    # Handle data formats
                    if isinstance(data, np.ndarray):
                        if data.dtype == object:  # Ragged array
                            min_length = min(len(x) for x in data)
                            train_loss = np.array([x[:min_length] for x in data])
                        else:
                            train_loss = data
                            
                        # Compute metrics
                        grad_norms = np.abs(np.gradient(train_loss))
                        gradient_data[lr][model][bs].append(grad_norms)
                        
                        d2_loss = np.gradient(np.gradient(train_loss))
                        sharpness_data[lr][model][bs].append(np.max(np.abs(d2_loss)))
                        
                print(f"✅ {model}: Processed {len(fold_files)} folds")
                
            except Exception as e:
                print(f"❌ Error in {model_dir.name}: {str(e)}")
                continue

# Plotting function for gradient norms with learning rate rows
def plot_gradient_norms_by_lr(gradient_data):
    fig, axes = plt.subplots(len(learning_rates), 1, figsize=(14, 6*len(learning_rates)))
    
    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        for model in model_names:
            for bs in batch_sizes:
                values = gradient_data[lr][model][bs]
                if values:
                    # Pad and average
                    max_len = max(len(v) for v in values)
                    padded = [np.pad(v, (0, max_len-len(v)), 'constant', constant_values=np.nan) 
                             for v in values]
                    mean = np.nanmean(padded, axis=0)
                    ax.plot(mean, 
                           color=model_colors[model],
                           linestyle=batch_linestyles[bs],
                           linewidth=2 if 'GGAR' in model else 1,
                           label=f"{model} ({bs})")  # Added label for legend
        
        ax.set_title(f'Gradient Norms | Learning Rate {lr.replace("_", ".")}', fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('|∂Loss/∂θ|', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})  # Added legend
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norms_by_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plotting function for sharpness boxplots with learning rate rows
def plot_sharpness_by_lr(sharpness_data):
    fig, axes = plt.subplots(len(learning_rates), 1, figsize=(14, 6*len(learning_rates)))
    
    # Create consistent color mapping for boxplots
    boxplot_colors = []
    boxplot_labels = []
    for model in model_names:
        for bs in batch_sizes:
            boxplot_colors.append(model_colors[model])
            boxplot_labels.append(f"{model}\n{bs}")
    
    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        data_to_plot = []
        for model in model_names:
            for bs in batch_sizes:
                values = sharpness_data[lr][model][bs]
                if values:
                    data_to_plot.append(values)

        # Create boxplot with consistent colors
        boxplot = ax.boxplot(data_to_plot, patch_artist=True)
        
        # Apply colors to boxes
        for patch, color in zip(boxplot['boxes'], boxplot_colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
        
        ax.set_xticklabels(boxplot_labels[:len(data_to_plot)], rotation=45)
        ax.set_title(f'Solution Sharpness | Learning Rate {lr.replace("_", ".")}', fontsize=14)
        ax.set_ylabel('|∂²Loss/∂θ²|', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Create unified legend
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=model) 
                      for model, color in model_colors.items()]
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpness_by_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_gradient_norms_by_lr(gradient_data)
plot_sharpness_by_lr(sharpness_data)

print(f"\nAll results saved to: {output_dir}")