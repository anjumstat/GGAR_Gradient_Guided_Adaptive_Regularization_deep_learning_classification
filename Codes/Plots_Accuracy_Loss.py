# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:50:59 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:01:17 2025

@author: H.A.R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration - Updated output directory
BASE_DIR = r'E:\GGAR\results\lr_0_0001_bs_32'

# Add 'MLP' model
METHODS = {
    'GGAR': {'dir': 'results_GGAR'},
    'AdaptiveL1L2': {'dir': 'results_Adaptive'},
    'FixedL1': {'dir': 'results_FixedL1'},
    'FixedL2': {'dir': 'results_FixedL2'},
    'ElasticNet': {'dir': 'results_ElasticNet'},
    'MLP': {'dir': 'results_MLP'}
}

# Updated color scheme
COLORS = {
    'GGAR':        '#E6194B',  # Bright Red (solid)
    'AdaptiveL1L2': '#3CB44B',  # Green (dashed)
    'FixedL1':     '#4363D8',  # Blue (dotted)
    'FixedL2':     '#F58231',  # Orange (dash-dot)
    'ElasticNet':  '#911EB4',  # Purple (solid, thicker)
    'MLP':         '#42D4F4',  # Cyan (solid, thin)
}


def process_history(method):
    """Process training history with proper error handling and padding"""
    method_dir = os.path.join(BASE_DIR, METHODS[method]['dir'])
    val_accs, val_losses = [], []
    train_accs, train_losses = [], []
    max_epochs = 0
    
    for fold in range(1, 11):
        try:
            val_acc = np.load(os.path.join(method_dir, f'fold{fold}_val_accuracy.npy'))
            val_loss = np.load(os.path.join(method_dir, f'fold{fold}_val_loss.npy'))
            train_acc = np.load(os.path.join(method_dir, f'fold{fold}_accuracy.npy'))
            train_loss = np.load(os.path.join(method_dir, f'fold{fold}_loss.npy'))
            
            current_epochs = len(val_acc)
            max_epochs = max(max_epochs, current_epochs)
            
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
        except Exception as e:
            print(f"Error processing {method} fold {fold}: {str(e)}")
            continue
    
    # Padding with edge values for consistent length
    def pad_sequences(sequences, target_length):
        return [np.pad(s, (0, target_length - len(s)), mode='edge') for s in sequences]
    
    padded_val_accs = pad_sequences(val_accs, max_epochs)
    padded_val_losses = pad_sequences(val_losses, max_epochs)
    padded_train_accs = pad_sequences(train_accs, max_epochs)
    padded_train_losses = pad_sequences(train_losses, max_epochs)

    return {
        'val_acc': np.mean(padded_val_accs, axis=0),
        'val_loss': np.mean(padded_val_losses, axis=0),
        'train_acc': np.mean(padded_train_accs, axis=0),
        'train_loss': np.mean(padded_train_losses, axis=0),
        'std_val_acc': np.std(padded_val_accs, axis=0),
        'std_val_loss': np.std(padded_val_losses, axis=0),
        'std_train_acc': np.std(padded_train_accs, axis=0),
        'std_train_loss': np.std(padded_train_losses, axis=0),
        'max_epochs': max_epochs
    }

# Create output directory if not exists
os.makedirs(BASE_DIR, exist_ok=True)

# 1. Training Dynamics Visualization
history_data = {method: process_history(method) for method in METHODS}

plt.figure(figsize=(15, 10))
metrics = ['val_acc', 'val_loss', 'train_acc', 'train_loss']
titles = ['Validation Accuracy', 'Validation Loss', 'Training Accuracy', 'Training Loss']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    plt.subplot(2, 2, idx + 1)
    for method, data in history_data.items():
        epochs = range(1, data['max_epochs'] + 1)
        mean = data[metric]
        std = data[f'std_{metric}']
        
        plt.plot(epochs, mean, label=method, color=COLORS[method])
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=COLORS[method])
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(title.split()[-1])
    if idx == 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add main title to the entire figure
plt.suptitle('Validation & Training  Accuracy - Validation & Training Loss Across Models\n Batch Size 32 - LR - 0.0001', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(BASE_DIR, 'training_dynamics.png'), bbox_inches='tight')
plt.close()

print(f"Training dynamics plot saved to: {os.path.join(BASE_DIR, 'training_dynamics.png')}")
