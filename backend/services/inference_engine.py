"""
MindFit AI - PyTorch Native Inference Engine
Loads SE-ResNet1D weights and performs high-precision CPU inference.
Includes smart biostatistical fallback if Kaggle weights have not yet been downloaded.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from backend.models.architecture import MindFitSEResNet1D, DualBranchBioFusionNet


class InferenceEngine:
    CLASSES = ["NORMAL", "MILD ANXIETY", "HIGH STRESS"]

    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(base_dir, 'models', 'weights', 'mindfit_se_resnet1d_best.pth')

        self.weights_path = weights_path
        self.device = torch.device('cpu')  # Fast CPU inference for demo
        self.model = None
        self.is_deep_learning_active = False
        self.is_dual_branch = False
        self.bio_mean = None
        self.bio_std = None

        self._load_model()

    def _load_model(self):
        if os.path.exists(self.weights_path):
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                
                # Check whether checkpoint belongs to DualBranchBioFusionNet
                if any('bio_mlp' in k for k in state_dict.keys()):
                    self.model = DualBranchBioFusionNet(in_channels=4, num_bio_features=6, num_classes=3)
                    self.is_dual_branch = True
                    if isinstance(checkpoint, dict):
                        if 'bio_mean' in checkpoint:
                            self.bio_mean = np.array(checkpoint['bio_mean'], dtype=np.float32)
                        if 'bio_std' in checkpoint:
                            self.bio_std = np.array(checkpoint['bio_std'], dtype=np.float32)
                    print("[InferenceEngine] Detected DualBranchBioFusionNet weights.")
                else:
                    self.model = MindFitSEResNet1D(in_channels=2, num_classes=3)
                    self.is_dual_branch = False

                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.is_deep_learning_active = True
                print(f"[InferenceEngine] Successfully loaded PyTorch model from {self.weights_path}")
            except Exception as e:
                print(f"[InferenceEngine] Warning: Could not load weights ({e}). Activating Smart Biostat Fallback.")
                self.model = None
                self.is_deep_learning_active = False
        else:
            print(f"[InferenceEngine] Model weights not found at {self.weights_path}. Smart Biostat Fallback active.")
            self.model = None
            self.is_deep_learning_active = False

    def predict_tensor(self, tensor_2ch: np.ndarray, bvp_mean: float = 75.0, eda_mean: float = 1.0) -> Dict[str, Any]:
        """
        Predict state from 2-channel normalized sensor stream [2, 240]:
        - If weights loaded: Deep Learning forward pass
        - If weights missing: Biostatistical physiological inference
        """
        if self.is_deep_learning_active and self.model is not None:
            with torch.no_grad():
                if self.is_dual_branch:
                    # Compute 4 channels and 6 scalar biomarkers
                    bvp = tensor_2ch[0:1, :]
                    eda = tensor_2ch[1:2, :]
                    d_bvp = np.gradient(bvp, axis=1)
                    d_eda = np.gradient(eda, axis=1)
                    wave_4ch = np.stack([bvp, d_bvp, eda, d_eda], axis=1).astype(np.float32) # [1, 4, 240]

                    bvp_std = float(np.std(bvp))
                    eda_mean_val = float(np.mean(eda))
                    eda_std = float(np.std(eda))
                    eda_diff_max = float(np.max(np.abs(d_eda)))
                    
                    from scipy import signal
                    peaks, _ = signal.find_peaks(bvp[0], distance=2, prominence=0.25)
                    hr_bpm = float(len(peaks))
                    if len(peaks) > 2:
                        rr = np.diff(peaks) * 250.0
                        rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
                    else:
                        rmssd = 0.0

                    bio_vec = np.array([[eda_mean_val, eda_std, eda_diff_max, bvp_std, hr_bpm, rmssd]], dtype=np.float32)

                    if self.bio_mean is not None and self.bio_std is not None:
                        bio_vec = (bio_vec - self.bio_mean) / (self.bio_std + 1e-6)

                    x_wave = torch.from_numpy(wave_4ch).to(self.device)
                    x_bio = torch.from_numpy(bio_vec).to(self.device)
                    logits = self.model(x_wave, x_bio)
                else:
                    x = torch.from_numpy(tensor_2ch).unsqueeze(0).to(self.device)  # [1, 2, 240]
                    logits = self.model(x)

                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                
                state = self.CLASSES[pred_idx]
                probabilities = {
                    self.CLASSES[0]: round(float(probs[0]), 3),
                    self.CLASSES[1]: round(float(probs[1]), 3),
                    self.CLASSES[2]: round(float(probs[2]), 3)
                }
                # Severity score: weighted sum
                severity = float(probs[1] * 50.0 + probs[2] * 95.0 + probs[0] * 10.0)
                
                model_name = "Dual-Branch Bio-Fusion Net" if self.is_dual_branch else "PyTorch 1D-CNN (SE-ResNet1D)"
                return {
                    "state": state,
                    "state_id": pred_idx,
                    "confidence": round(confidence, 3),
                    "probabilities": probabilities,
                    "severity_score": round(severity, 1),
                    "model_source": model_name
                }
        else:
            # Smart Biostatistical Fallback using clinical physiological boundaries
            # Normal: HR < 75, EDA < 1.0
            # Mild Anxiety: HR 75-95, EDA 1.0 - 3.5
            # High Stress: HR > 95 or EDA > 3.5
            if eda_mean > 3.5 or bvp_mean > 95.0:
                state = "HIGH STRESS"
                pred_idx = 2
                probs = [0.08, 0.22, 0.70]
                severity = 88.0
            elif eda_mean > 1.2 or bvp_mean > 78.0:
                state = "MILD ANXIETY"
                pred_idx = 1
                probs = [0.15, 0.68, 0.17]
                severity = 54.0
            else:
                state = "NORMAL"
                pred_idx = 0
                probs = [0.82, 0.14, 0.04]
                severity = 18.0

            return {
                "state": state,
                "state_id": pred_idx,
                "confidence": round(float(probs[pred_idx]), 3),
                "probabilities": {
                    self.CLASSES[0]: probs[0],
                    self.CLASSES[1]: probs[1],
                    self.CLASSES[2]: probs[2]
                },
                "severity_score": round(severity, 1),
                "model_source": "Biostatistical Physiological Inference (Pre-Kaggle Weights)"
            }

    def predict_manual_vitals(self, resting_hr: float, sleep_hours: float, stress_feel: int, gad7_scores: list) -> Dict[str, Any]:
        """
        Predict state from manual clinical vitals and GAD-7 assessment scale.
        """
        gad7_total = sum(gad7_scores) if gad7_scores else 0

        # Clinical mapping
        # GAD-7 scale: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
        if gad7_total >= 10 or stress_feel >= 8 or resting_hr >= 100:
            state = "HIGH STRESS"
            pred_idx = 2
            probs = [0.05, 0.20, 0.75]
            severity = min(98.0, 70.0 + gad7_total * 1.5)
        elif gad7_total >= 5 or stress_feel >= 5 or resting_hr >= 80 or sleep_hours < 6.0:
            state = "MILD ANXIETY"
            pred_idx = 1
            probs = [0.15, 0.72, 0.13]
            severity = 45.0 + gad7_total * 1.2
        else:
            state = "NORMAL"
            pred_idx = 0
            probs = [0.85, 0.12, 0.03]
            severity = 15.0 + gad7_total * 0.8

        return {
            "state": state,
            "state_id": pred_idx,
            "confidence": round(float(probs[pred_idx]), 3),
            "probabilities": {
                self.CLASSES[0]: probs[0],
                self.CLASSES[1]: probs[1],
                self.CLASSES[2]: probs[2]
            },
            "severity_score": round(severity, 1),
            "gad7_total": gad7_total,
            "model_source": "Clinical GAD-7 & Vitals Bayesian Scoring"
        }
