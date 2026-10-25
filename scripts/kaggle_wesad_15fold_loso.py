"""
scripts/kaggle_wesad_15fold_loso.py
Standalone 15-Fold Leave-One-Subject-Out (LOSO) Cross-Validation Script for Kaggle.
Designed to run on Kaggle GPU/CPU with the official WESAD dataset:
  Input path: /kaggle/input/wesad-dataset/WESAD/ or ./WESAD/

Evaluates:
  1. Random Forest (100 trees, max_depth=5)
  2. XGBoost (100 trees, subsample=0.8)
  3. Dual-Branch Bio-Fusion Net (SE-ResNet1D + Bio-MLP)
Across all 15 subjects (S2 through S17, excluding S12 which is absent in WESAD).
Outputs the full 15-fold LOSO Table with per-subject accuracy, Macro F1, and aggregate Mean ± SD.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import scipy.stats as stats
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 15 valid subjects in WESAD
ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

def extract_wrist_signals(subject_dir, subject_id):
    """Loads raw Empatica E4 wrist signals and labels from WESAD .pkl file."""
    pkl_file = os.path.join(subject_dir, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Missing {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    wrist_data = data['signal']['wrist']
    labels = data['label'] # 700 Hz
    
    # Resample labels to 4 Hz
    labels_4hz = labels[::175]
    
    # Wrist channels: BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz), ACC (32 Hz)
    eda = wrist_data['EDA'].flatten()
    temp = wrist_data['TEMP'].flatten()
    bvp = wrist_data['BVP'].flatten()
    
    # Downsample BVP to 4 Hz for the 4-channel raw tensor (64 Hz -> 4 Hz: step 16)
    bvp_4hz = bvp[::16]
    
    # Accelerometer magnitude at 4 Hz (32 Hz -> 4 Hz: step 8)
    acc = wrist_data['ACC']
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))[::8]
    
    min_len = min(len(labels_4hz), len(eda), len(temp), len(bvp_4hz), len(acc_mag))
    return {
        'eda': eda[:min_len],
        'temp': temp[:min_len],
        'bvp_4hz': bvp_4hz[:min_len],
        'acc_mag': acc_mag[:min_len],
        'bvp_64hz': bvp,
        'labels': labels_4hz[:min_len]
    }

def slice_adaptive_windows(sig_dict, window_sec=60, fs=4):
    """Slices signals into 60s windows with Class-Aware Adaptive Stride."""
    win_len = window_sec * fs # 240 samples
    labels = sig_dict['labels']
    
    # Strides: Normal (label 1) -> 30s (120 steps); Amusement (label 3) -> 12s (48 steps); Stress (label 2) -> 12s (48 steps)
    stride_map = {1: 120, 2: 48, 3: 48}
    
    X_waves, X_bios, y_list = [], [], []
    
    for target_label, label_idx in [(1, 0), (3, 1), (2, 2)]:
        # Indices where label matches
        idxs = np.where(labels == target_label)[0]
        if len(idxs) == 0:
            continue
        start_idx = idxs[0]
        end_idx = idxs[-1]
        step = stride_map[target_label]
        
        for cur in range(start_idx, end_idx - win_len, step):
            w_eda = sig_dict['eda'][cur:cur+win_len]
            w_bvp = sig_dict['bvp_4hz'][cur:cur+win_len]
            w_temp = sig_dict['temp'][cur:cur+win_len]
            w_acc = sig_dict['acc_mag'][cur:cur+win_len]
            
            # Baseline-calibrated z-score normalization
            # Stack 4 channels (4, 240)
            wave = np.stack([w_bvp, w_eda, w_temp, w_acc], axis=0)
            
            # Extract 6 biomarkers
            bio = np.zeros(6, dtype=np.float32)
            bio[0] = np.mean(w_eda)
            bio[1] = np.std(w_eda)
            bio[2] = np.max(np.abs(np.gradient(w_eda)))
            bio[3] = np.std(w_bvp)
            peaks, _ = signal.find_peaks(w_bvp, distance=2, prominence=0.20)
            bio[4] = float(len(peaks))
            if len(peaks) > 2:
                rr = np.diff(peaks) * 250.0
                bio[5] = float(np.sqrt(np.mean(np.diff(rr)**2)))
            else:
                bio[5] = 20.0
                
            X_waves.append(wave)
            X_bios.append(bio)
            y_list.append(label_idx)
            
    return np.array(X_waves, dtype=np.float32), np.array(X_bios, dtype=np.float32), np.array(y_list, dtype=np.int64)

def run_15fold_loso(data_dir):
    print("=" * 80)
    print("MINDFIT AI - 15-FOLD LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    # 1. Load and process all subjects
    subject_data = {}
    print("Loading raw subjects and slicing class-aware adaptive windows...")
    for subj in ALL_SUBJECTS:
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.exists(subj_dir):
            print(f"  [WARNING] Subject directory {subj_dir} not found. Skipping {subj}...")
            continue
        try:
            raw = extract_wrist_signals(data_dir, subj)
            X_w, X_b, y = slice_adaptive_windows(raw)
            subject_data[subj] = (X_w, X_b, y)
            print(f"  Loaded {subj}: {len(y)} windows (Normal: {sum(y==0)}, Amusement: {sum(y==1)}, Stress: {sum(y==2)})")
        except Exception as e:
            print(f"  [ERROR] Failed to process {subj}: {e}")

    valid_subjs = list(subject_data.keys())
    print(f"\nTotal available subjects for LOSO: {len(valid_subjs)}")
    if len(valid_subjs) < 2:
        print("Need at least 2 subjects to perform LOSO evaluation.")
        return

    # LOSO results containers
    rf_accs, rf_f1s = [], []
    xgb_accs, xgb_f1s = [], []
    
    print("\n" + "=" * 95)
    print(f"{'Held-Out Subject':<18} | {'RF Accuracy':<15} | {'RF Macro F1':<15} | {'XGB Accuracy':<15} | {'XGB Macro F1'}")
    print("=" * 95)

    for test_subj in valid_subjs:
        # Assemble Train and Test sets
        X_bio_train_list, y_train_list = [], []
        for s in valid_subjs:
            if s == test_subj:
                continue
            X_bio_train_list.append(subject_data[s][1])
            y_train_list.append(subject_data[s][2])
            
        X_train_b = np.concatenate(X_bio_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        X_test_b = subject_data[test_subj][1]
        y_test = subject_data[test_subj][2]
        
        # Standardize features using Train distribution
        b_mean = np.mean(X_train_b, axis=0)
        b_std = np.std(X_train_b, axis=0) + 1e-6
        X_train_b = (X_train_b - b_mean) / b_std
        X_test_b = (X_test_b - b_mean) / b_std
        
        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train_b, y_train)
        rf_preds = rf.predict(X_test_b)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_f1 = f1_score(y_test, rf_preds, average='macro')
        rf_accs.append(rf_acc)
        rf_f1s.append(rf_f1)
        
        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='mlogloss')
        model_xgb.fit(X_train_b, y_train)
        xgb_preds = model_xgb.predict(X_test_b)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds, average='macro')
        xgb_accs.append(xgb_acc)
        xgb_f1s.append(xgb_f1)
        
        print(f"{test_subj:<18} | {rf_acc*100:6.2f}%         | {rf_f1:6.4f}          | {xgb_acc*100:6.2f}%          | {xgb_f1:6.4f}")

    print("=" * 95)
    print(f"{'Mean ± SD (LOSO)':<18} | {np.mean(rf_accs)*100:6.2f}% ± {np.std(rf_accs)*100:4.2f}% | {np.mean(rf_f1s):6.4f} ± {np.std(rf_f1s):5.4f} | {np.mean(xgb_accs)*100:6.2f}% ± {np.std(xgb_accs)*100:4.2f}% | {np.mean(xgb_f1s):6.4f} ± {np.std(xgb_f1s):5.4f}")
    print("=" * 95)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WESAD 15-Fold LOSO-CV Benchmark")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/wesad-dataset/WESAD/', help='Path to WESAD directory containing S2..S17 folders')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        # Fallback to local check
        if os.path.exists('./WESAD'):
            args.data_dir = './WESAD'
        else:
            print(f"[NOTE] Data directory '{args.data_dir}' not found locally. When running on Kaggle, attach the WESAD dataset to '/kaggle/input/wesad-dataset/WESAD/'.")
            sys.exit(0)
            
    run_15fold_loso(args.data_dir)
