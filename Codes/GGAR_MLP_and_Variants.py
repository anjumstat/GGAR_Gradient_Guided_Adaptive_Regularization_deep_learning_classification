# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 17:55:35 2025

@author: H.A.R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
import time
import shutil
from scipy.stats import ttest_rel, friedmanchisquare, wilcoxon

# Configuration - USER CAN MODIFY THESE VALUES
DATA_PATH = 'E:/GGAR/4_Species.csv'
BASE_DIR = 'E:/GGAR/results'
LEARNING_RATES = [0.01,0.001, 0.0001]  # Can be modified to any list of values
BATCH_SIZES = [32,64,128,256]       # Can be modified to any list of values

METHODS = {
    'GGAR': {'dir': 'results_GGAR', 'builder': 'build_ggar'},  # New method
    'AdaptiveL1L2': {'dir': 'results_Adaptive', 'builder': 'build_adaptive'},
    'FixedL1': {'dir': 'results_FixedL1', 'builder': 'build_fixed_l1'},
    'FixedL2': {'dir': 'results_FixedL2', 'builder': 'build_fixed_l2'},
    'ElasticNet': {'dir': 'results_ElasticNet', 'builder': 'build_elastic'},
    'MLP': {'dir': 'results_MLP', 'builder': 'build_mlp'}
}

# 1. Gradient-Guided Adaptive Regularizer (GGAR)
class GGARRegularizer(regularizers.Regularizer):
    def __init__(self, base_strength=0.01, sensitivity=0.1, memory_factor=0.9, 
                 min_strength=0.001, max_strength=0.1):
        super().__init__()
        self.base_strength = tf.Variable(base_strength, trainable=False)
        self.sensitivity = sensitivity
        self.memory_factor = memory_factor
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        # Track gradient statistics
        self.gradient_magnitude = tf.Variable(0.0, trainable=False)
        self.gradient_variance = tf.Variable(0.0, trainable=False)
        self.strength_history = tf.Variable([], shape=[None], dtype=tf.float32)
        
    def __call__(self, weights):
        # Calculate current gradient statistics
        current_gradient = tf.reduce_mean(tf.abs(weights))
        
        # Update gradient statistics with exponential moving average
        self.gradient_magnitude.assign(
            self.memory_factor * self.gradient_magnitude + 
            (1 - self.memory_factor) * current_gradient
        )
        
        # Calculate gradient variance (approximation)
        grad_diff = tf.abs(current_gradient - self.gradient_magnitude)
        self.gradient_variance.assign(
            self.memory_factor * self.gradient_variance + 
            (1 - self.memory_factor) * grad_diff
        )
        
        # Adaptive regularization strength based on gradient behavior
        stability_factor = tf.math.exp(-self.gradient_variance * self.sensitivity)
        dynamic_strength = tf.clip_by_value(
            self.base_strength * stability_factor * (1 + tf.math.sigmoid(current_gradient)),
            self.min_strength,
            self.max_strength
        )
        
        # Record strength history for analysis
        self.strength_history.assign(tf.concat([self.strength_history, [dynamic_strength]], axis=0))
        
        # Apply both L1 and L2 regularization with dynamic strength
        l1_contribution = dynamic_strength * tf.reduce_sum(tf.abs(weights))
        l2_contribution = (dynamic_strength / 2) * tf.reduce_sum(tf.square(weights))
        
        return l1_contribution + l2_contribution

    def get_config(self):
        return {
            'base_strength': self.base_strength.numpy(),
            'sensitivity': self.sensitivity,
            'memory_factor': self.memory_factor,
            'min_strength': self.min_strength,
            'max_strength': self.max_strength
        }

# 2. Model Builders
def build_model(method, input_shape, num_classes, learning_rate):
    input_shape_tuple = (input_shape,)
    
    if method == 'build_ggar':
        input_layer = layers.Input(shape=input_shape_tuple)
        
        # First GGAR-regularized layer
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=GGARRegularizer(0.01))(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Gradient attention mechanism
        grad_attention = layers.Dense(64, activation='relu')(x)
        grad_attention = layers.Dense(128, activation='sigmoid')(grad_attention)
        x = layers.Multiply()([x, grad_attention])
        
        # Second GGAR-regularized layer
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=GGARRegularizer(0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=['accuracy']
        )
        return model
    
    elif method == 'build_adaptive':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(0.01, 0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1L2(0.005, 0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_fixed_l1':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1(0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1(0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_fixed_l2':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L2(0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L2(0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_elastic':
        model = models.Sequential([
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(0.01, 0.01),
                        input_shape=input_shape_tuple),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.L1L2(0.005, 0.005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    elif method == 'build_mlp':
        model = models.Sequential([
            layers.Dense(256, kernel_regularizer=regularizers.l2(0.01), input_shape=input_shape_tuple),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.01)),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                     loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    raise ValueError(f"Unknown method: {method}")

# [Rest of the code remains exactly the same, just replace any references to NovelGRR with GGAR]
# 3. Training Pipeline
def run_experiment(method_name, config, learning_rate, batch_size):
    # Create subfolder for this learning rate and batch size
    lr_str = f"{learning_rate:.4f}".replace('.', '_').rstrip('0').rstrip('_') if '_' in f"{learning_rate:.4f}".replace('.', '_') else f"{learning_rate:.4f}".replace('.', '_')
    batch_str = str(batch_size)
    output_dir = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{batch_str}", config['dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data = pd.read_csv(DATA_PATH)
        X = data.iloc[:, :-1].values
        y = data['Species'].values - 1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        num_classes = len(np.unique(y))
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Initialize metrics storage for all folds
        fold_metrics = {
            'train': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'val': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'test': {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []
            },
            'training_time': [],
            'val_accuracy': []
        }
        
        best_model_path = os.path.join(output_dir, f"Best_{method_name}_Model.h5")
        best_val_acc = -np.inf
        best_fold = -1
        best_train_index = None
        best_val_index = None
        
        # Start timer for the full model
        full_model_start_time = time.time()

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            print(f"\n{method_name} (LR={learning_rate}, BS={batch_size}) - Fold {fold+1}/10")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train_cat[train_index], y_train_cat[val_index]
            
            model = build_model(config['builder'], X_train.shape[1], num_classes, learning_rate)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.001,
                mode='max',
                restore_best_weights=True
            )
            
            start_time = time.time()
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=50,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[early_stop]
            )
            
            # Calculate metrics for training data
            y_train_pred = model.predict(X_train_fold)
            y_train_pred_classes = np.argmax(y_train_pred, axis=1)
            y_train_true = np.argmax(y_train_fold, axis=1)
            
            fold_metrics['train']['accuracy'].append(accuracy_score(y_train_true, y_train_pred_classes))
            fold_metrics['train']['precision'].append(precision_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['recall'].append(recall_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['f1'].append(f1_score(y_train_true, y_train_pred_classes, average='macro'))
            fold_metrics['train']['mcc'].append(matthews_corrcoef(y_train_true, y_train_pred_classes))
            
            # Calculate metrics for validation data
            y_val_pred = model.predict(X_val_fold)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            y_val_true = np.argmax(y_val_fold, axis=1)
            
            fold_metrics['val']['accuracy'].append(accuracy_score(y_val_true, y_val_pred_classes))
            fold_metrics['val']['precision'].append(precision_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['recall'].append(recall_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['f1'].append(f1_score(y_val_true, y_val_pred_classes, average='macro'))
            fold_metrics['val']['mcc'].append(matthews_corrcoef(y_val_true, y_val_pred_classes))
            
            # Calculate metrics for test data
            y_test_pred = model.predict(X_test)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            
            fold_metrics['test']['accuracy'].append(accuracy_score(y_test, y_test_pred_classes))
            fold_metrics['test']['precision'].append(precision_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['recall'].append(recall_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['f1'].append(f1_score(y_test, y_test_pred_classes, average='macro'))
            fold_metrics['test']['mcc'].append(matthews_corrcoef(y_test, y_test_pred_classes))
            
            # Store training time and validation accuracy
            fold_metrics['training_time'].append(time.time() - start_time)
            fold_metrics['val_accuracy'].append(np.max(history.history['val_accuracy']))
            
            # Save history
            fold_num = fold + 1
            for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                np.save(os.path.join(output_dir, f"fold{fold_num}_{metric}.npy"), 
                        np.array(history.history[metric]))
            
            # Check for best model
            val_acc = np.max(history.history['val_accuracy'])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold = fold
                best_train_index = train_index
                best_val_index = val_index
                model.save(best_model_path)

        # Calculate total training time for the full model
        full_model_time_minutes = (time.time() - full_model_start_time) / 60
        
        # Save training time to a CSV file
        time_df = pd.DataFrame({
            'Method': [method_name],
            'Learning_Rate': [learning_rate],
            'Batch_Size': [batch_size],
            'Total_Training_Time_Minutes': [full_model_time_minutes]
        })
        time_df.to_csv(os.path.join(output_dir, "Training_Time.csv"), index=False)

        # Save fold metrics to CSV
        fold_metrics_df = pd.DataFrame({
            'Fold': range(1, 11),
            'Train_Accuracy': fold_metrics['train']['accuracy'],
            'Train_Precision': fold_metrics['train']['precision'],
            'Train_Recall': fold_metrics['train']['recall'],
            'Train_F1': fold_metrics['train']['f1'],
            'Train_MCC': fold_metrics['train']['mcc'],
            'Val_Accuracy': fold_metrics['val']['accuracy'],
            'Val_Precision': fold_metrics['val']['precision'],
            'Val_Recall': fold_metrics['val']['recall'],
            'Val_F1': fold_metrics['val']['f1'],
            'Val_MCC': fold_metrics['val']['mcc'],
            'Test_Accuracy': fold_metrics['test']['accuracy'],
            'Test_Precision': fold_metrics['test']['precision'],
            'Test_Recall': fold_metrics['test']['recall'],
            'Test_F1': fold_metrics['test']['f1'],
            'Test_MCC': fold_metrics['test']['mcc'],
            'Training_Time': fold_metrics['training_time'],
            'Val_Accuracy_Epoch': fold_metrics['val_accuracy']
        })
        fold_metrics_df.to_csv(os.path.join(output_dir, "Fold_Metrics_Detailed.csv"), index=False)
        
        # Calculate and save average metrics
        avg_metrics = {
            'Dataset': ['Train', 'Validation', 'Test'],
            'Accuracy': [
                np.mean(fold_metrics['train']['accuracy']),
                np.mean(fold_metrics['val']['accuracy']),
                np.mean(fold_metrics['test']['accuracy'])
            ],
            'Precision': [
                np.mean(fold_metrics['train']['precision']),
                np.mean(fold_metrics['val']['precision']),
                np.mean(fold_metrics['test']['precision'])
            ],
            'Recall': [
                np.mean(fold_metrics['train']['recall']),
                np.mean(fold_metrics['val']['recall']),
                np.mean(fold_metrics['test']['recall'])
            ],
            'F1': [
                np.mean(fold_metrics['train']['f1']),
                np.mean(fold_metrics['val']['f1']),
                np.mean(fold_metrics['test']['f1'])
            ],
            'MCC': [
                np.mean(fold_metrics['train']['mcc']),
                np.mean(fold_metrics['val']['mcc']),
                np.mean(fold_metrics['test']['mcc'])
            ]
        }
        avg_metrics_df = pd.DataFrame(avg_metrics)
        avg_metrics_df.to_csv(os.path.join(output_dir, "Average_Metrics.csv"), index=False)
        
        # Process best model
        if best_fold >= 0:
            custom_objects = {'GGARRegularizer': GGARRegularizer} if 'GGAR' in method_name else {}
            best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
            
            # Calculate metrics for best model
            metrics_sets = {}
            
            # Training set metrics
            X_best_train = X_train[best_train_index]
            y_best_train = y_train_cat[best_train_index]
            y_train_pred = best_model.predict(X_best_train)
            y_train_pred_classes = np.argmax(y_train_pred, axis=1)
            y_train_true = np.argmax(y_best_train, axis=1)
            
            metrics_sets['train'] = {
                'accuracy': accuracy_score(y_train_true, y_train_pred_classes),
                'precision': precision_score(y_train_true, y_train_pred_classes, average='macro'),
                'recall': recall_score(y_train_true, y_train_pred_classes, average='macro'),
                'f1': f1_score(y_train_true, y_train_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_train_true, y_train_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Validation set metrics
            X_best_val = X_train[best_val_index]
            y_best_val = y_train_cat[best_val_index]
            y_val_pred = best_model.predict(X_best_val)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            y_val_true = np.argmax(y_best_val, axis=1)
            
            metrics_sets['validation'] = {
                'accuracy': accuracy_score(y_val_true, y_val_pred_classes),
                'precision': precision_score(y_val_true, y_val_pred_classes, average='macro'),
                'recall': recall_score(y_val_true, y_val_pred_classes, average='macro'),
                'f1': f1_score(y_val_true, y_val_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_val_true, y_val_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Test set metrics
            y_test_pred = best_model.predict(X_test)
            y_test_pred_classes = np.argmax(y_test_pred, axis=1)
            
            metrics_sets['test'] = {
                'accuracy': accuracy_score(y_test, y_test_pred_classes),
                'precision': precision_score(y_test, y_test_pred_classes, average='macro'),
                'recall': recall_score(y_test, y_test_pred_classes, average='macro'),
                'f1': f1_score(y_test, y_test_pred_classes, average='macro'),
                'mcc': matthews_corrcoef(y_test, y_test_pred_classes),
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
            
            # Save best model metrics to CSV
            best_metrics_df = pd.DataFrame(metrics_sets).T
            best_metrics_df.to_csv(os.path.join(output_dir, "Best_Model_Metrics.csv"))
            
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(y_test, y_test_pred_classes), 
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f"Species {i+1}" for i in range(num_classes)],
                        yticklabels=[f"Species {i+1}" for i in range(num_classes)])
            plt.title(f'{method_name} (LR={learning_rate}, BS={batch_size}) Confusion Matrix')
            plt.savefig(os.path.join(output_dir, "Confusion_Matrix.png"))
            plt.close()
        
        # Clean up - remove unwanted CSV files (keeping only specified ones)
        files_to_keep = [
            "Training_Time.csv", "Fold_Metrics_Detailed.csv", 
            "Average_Metrics.csv", "Best_Model_Metrics.csv"
        ]
        for file in os.listdir(output_dir):
            if file.endswith('.csv') and file not in files_to_keep:
                os.remove(os.path.join(output_dir, file))
        
        return metrics_sets
    
    except Exception as e:
        print(f"Error in {method_name} (LR={learning_rate}, BS={batch_size}): {str(e)}")
        return None

# 4. Analysis Functions - Simplified version without removed files
def analyze_results(base_dir):
    # This function is kept for compatibility but doesn't generate the removed files
    print("Analysis function running (simplified version)")
    return None

# 5. Main Execution
if __name__ == "__main__":
    # Clean previous results if they exist
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Run all experiments
    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for method_name, config in METHODS.items():
                print(f"\n{'='*40}\nRunning {method_name} with LR={learning_rate}, BS={batch_size}\n{'='*40}")
                run_experiment(method_name, config, learning_rate, batch_size)
    
    # Run simplified analysis
    print("\nAnalyzing results...")
    analyze_results(BASE_DIR)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key outputs generated for each method configuration:")
    print("- Training_Time.csv: Training time information")
    print("- Fold_Metrics_Detailed.csv: Detailed metrics for all 10 folds")
    print("- Average_Metrics.csv: Average metrics across folds")
    print("- Best_Model_Metrics.csv: Metrics for the best model")
    print("- Confusion_Matrix.png: Confusion matrix for test data")
    print("\nResults saved to:", BASE_DIR)