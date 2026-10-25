"""
MindFit AI - Signal Processing Service
Handles noise filtering, resampling, and window segmentation for physiological time-series:
- BVP / PPG Heart Rate Pulse: Butterworth Bandpass Filter (0.5 - 4.0 Hz)
- EDA (Electrodermal Activity): Butterworth Lowpass Filter (1.0 Hz)
- Uniform Resampling to 4 Hz (240 timesteps for 60s windows)
- Z-score Normalization
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Dict, Any, List, Optional


class SignalProcessor:
    def __init__(self, target_fs: float = 4.0, window_sec: int = 60):
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.seq_len = int(target_fs * window_sec)  # 240 samples

    def butter_bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 4.0, fs: float = 4.0, order: int = 3) -> np.ndarray:
        """Bandpass filter for heart pulse signal (0.5 - 4.0 Hz, 30-240 bpm)."""
        nyq = 0.5 * fs
        low = max(0.01, min(lowcut / nyq, 0.99))
        high = max(low + 0.01, min(highcut / nyq, 0.99))
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.filtfilt(b, a, data)
        except Exception:
            return data

    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float = 1.0, fs: float = 4.0, order: int = 3) -> np.ndarray:
        """Lowpass filter for EDA signal (attenuates motion artifacts above 1.0 Hz)."""
        nyq = 0.5 * fs
        normal_cutoff = max(0.01, min(cutoff / nyq, 0.99))
        try:
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            return signal.filtfilt(b, a, data)
        except Exception:
            return data

    def resample_signal(self, data: np.ndarray, orig_fs: float, target_fs: float = 4.0) -> np.ndarray:
        """Resample 1D signal to target sampling frequency."""
        if orig_fs == target_fs or len(data) == 0:
            return data
        target_len = int(len(data) * (target_fs / orig_fs))
        return signal.resample(data, target_len)

    def parse_wearable_csv(self, file_content: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Parse uploaded CSV from Apple Watch, Fitbit, or Empatica.
        Detects columns for Heart Rate/BVP and EDA with fuzzy keyword matching.
        """
        from io import StringIO
        df = pd.read_csv(StringIO(file_content))
        df.columns = [c.strip().lower() for c in df.columns]

        # 1. Resolve BVP / Heart Rate column
        bvp_col = None
        for c in df.columns:
            if any(term in c for term in ['bvp', 'ppg', 'pulse', 'heart_rate', 'hr', 'cardiac', 'rate']):
                bvp_col = c
                break

        # 2. Resolve EDA / Skin Conductance column
        eda_col = None
        for c in df.columns:
            if any(term in c for term in ['eda', 'gsr', 'conductance', 'skin', 'galvanic', 'microsiemens']):
                eda_col = c
                break

        # Fallback if specific naming isn't present
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if 'time' not in c and 'date' not in c]
        if bvp_col is None:
            bvp_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
        if eda_col is None:
            remaining = [c for c in numeric_cols if c != bvp_col]
            eda_col = remaining[0] if len(remaining) > 0 else bvp_col

        bvp_raw = df[bvp_col].to_numpy(dtype=np.float32)
        eda_raw = df[eda_col].to_numpy(dtype=np.float32)

        # Estimate sampling rate if timestamp exists
        fs = 4.0
        time_col = None
        for c in df.columns:
            if any(term in c for term in ['time', 'timestamp', 'datetime', 'sec']):
                time_col = c
                break

        if time_col is not None and len(df) > 1:
            try:
                t = pd.to_numeric(df[time_col], errors='coerce').dropna().to_numpy()
                if len(t) > 1:
                    dt = np.median(np.diff(t))
                    if dt > 0:
                        if dt > 50:  # ms
                            dt = dt / 1000.0
                        fs = max(1.0, min(1.0 / dt, 128.0))
            except Exception:
                fs = 4.0

        return bvp_raw, eda_raw, float(fs)

    def process_and_window(self, bvp_raw: np.ndarray, eda_raw: np.ndarray, orig_fs: float) -> Dict[str, Any]:
        """
        Process raw inputs into a clean 2-channel tensor suitable for 1D-CNN:
        - Resamples to 4 Hz
        - Butterworth filters
        - Windows to 240 samples
        - Returns preview data for frontend wave visualizer
        """
        # Store true unnormalized vitals before zero-centering
        raw_bvp_mean = float(np.mean(bvp_raw))
        raw_bvp_std = float(np.std(bvp_raw))
        raw_eda_mean = float(np.mean(eda_raw))
        raw_eda_std = float(np.std(eda_raw))

        # Resample to 4Hz
        bvp_resampled = self.resample_signal(bvp_raw, orig_fs, self.target_fs)
        eda_resampled = self.resample_signal(eda_raw, orig_fs, self.target_fs)

        # Apply digital filters
        bvp_filtered = self.butter_bandpass_filter(bvp_resampled, fs=self.target_fs)
        eda_filtered = self.butter_lowpass_filter(eda_resampled, fs=self.target_fs)

        # Ensure minimum length of seq_len (240 samples = 60s)
        if len(bvp_filtered) < self.seq_len:
            pad_len = self.seq_len - len(bvp_filtered)
            bvp_filtered = np.pad(bvp_filtered, (0, pad_len), mode='edge')
            eda_filtered = np.pad(eda_filtered, (0, pad_len), mode='edge')
            bvp_resampled = np.pad(bvp_resampled, (0, pad_len), mode='edge')
        else:
            bvp_filtered = bvp_filtered[-self.seq_len:]
            eda_filtered = eda_filtered[-self.seq_len:]
            bvp_resampled = bvp_resampled[-self.seq_len:]

        # Standard Z-score Normalization per channel for CNN input
        bvp_norm = (bvp_filtered - np.mean(bvp_filtered)) / (np.std(bvp_filtered) + 1e-6)
        eda_norm = (eda_filtered - np.mean(eda_filtered)) / (np.std(eda_filtered) + 1e-6)

        # Shape: [2, 240]
        tensor_input = np.stack([bvp_norm, eda_norm], axis=0).astype(np.float32)

        # Preview 100 points for frontend rendering
        preview_indices = np.linspace(0, self.seq_len - 1, 100).astype(int)

        # In frontend preview, reconstruct calibrated pulse waveform around baseline HR
        display_pulse = [round(float(raw_bvp_mean + bvp_norm[i] * 4.0), 1) for i in preview_indices]
        display_eda = [round(float(eda_resampled[i]), 3) for i in preview_indices]

        return {
            "tensor": tensor_input,
            "bvp_mean": raw_bvp_mean,
            "bvp_std": raw_bvp_std,
            "eda_mean": raw_eda_mean,
            "eda_std": raw_eda_std,
            "visual_preview": {
                "raw_wave": [round(float(v), 1) for v in bvp_resampled[preview_indices]],
                "filtered_bvp": display_pulse,
                "filtered_eda": display_eda,
                "timestamps": [f"{i*0.6:.1f}s" for i in range(100)]
            }
        }
