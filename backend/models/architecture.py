"""
MindFit AI - Deep Learning Architecture
Model: Multi-Scale Squeeze-and-Excitation Residual 1D-CNN (SE-ResNet1D)
Trained on Wearable Time-Series (BVP/HRV, EDA) for Anxiety & Stress Classification
Classes: 0: NORMAL, 1: MILD ANXIETY, 2: HIGH STRESS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class SqueezeExcitation1D(nn.Module):
    """
    1D Squeeze-and-Excitation Block.
    Dynamically recalculates channel-wise feature dependencies (e.g. giving higher
    attention weight to sympathetic EDA spikes or parasympathetic HRV drops).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        weights = self.fc(x).view(b, c, 1)
        return x * weights


class MultiScaleConv1D(nn.Module):
    """
    Multi-Scale Temporal Convolutional Block.
    Captures both fast high-frequency variations (BVP pulse) and slow galvanic skin response (EDA)
    using parallel kernel sizes (3, 7, 15).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(in_channels, remainder, kernel_size=15, padding=7),
            nn.BatchNorm1d(remainder),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch7(x), self.branch15(x)], dim=1)


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Squeeze-and-Excitation attention and skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SqueezeExcitation1D(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.gelu(out)


class MindFitSEResNet1D(nn.Module):
    """
    SOTA Multi-Scale SE-ResNet1D for Anxiety and Stress State Recognition.
    - Input shape: [Batch, Channels=2, Seq_len=240] (60s window at 4Hz sampling)
      Channel 0: Blood Volume Pulse / Normalized Heart Rate Waveform
      Channel 1: Electrodermal Activity (Skin Conductance)
    - Output shape: [Batch, 3] (Logits for NORMAL, MILD ANXIETY, HIGH STRESS)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem Layer with Multi-Scale temporal receptive fields
        self.stem = MultiScaleConv1D(in_channels, out_channels=32)

        # Deep Residual Feature Extractors
        self.layer1 = ResidualBlock1D(32, 64, stride=2)   # 240 -> 120 timesteps
        self.layer2 = ResidualBlock1D(64, 128, stride=2)  # 120 -> 60 timesteps
        self.layer3 = ResidualBlock1D(128, 256, stride=2) # 60 -> 30 timesteps

        # Global Pooling and Linear Classifier Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature representation for latent analysis or clustering."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.global_pool(x).flatten(1)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance between baseline states
    and acute stress/anxiety episodes.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DualBranchBioFusionNet(nn.Module):
    """
    SOTA Dual-Branch Bio-Fusion Network for Anxiety & Stress Classification:
    - Waveform Branch: 4-Channel Input [BVP, d(BVP)/dt, EDA, d(EDA)/dt] -> MultiScale Conv1D -> 3 SE-ResNet Blocks -> GlobalAvgPool (192-d)
    - Biomarker Branch: 6 Scalar Physiological Features -> Bio-MLP (32-d)
    - Cross-Feature Fusion Head: 224-d -> Dense(64) -> Output(3)
    """
    def __init__(self, in_channels: int = 4, num_bio_features: int = 6, num_classes: int = 3):
        super().__init__()
        # 1. Temporal Waveform Branch
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(16), nn.GELU())
        self.b7 = nn.Sequential(nn.Conv1d(in_channels, 16, kernel_size=7, padding=3, bias=False), nn.BatchNorm1d(16), nn.GELU())
        self.b15 = nn.Sequential(nn.Conv1d(in_channels, 16, kernel_size=15, padding=7, bias=False), nn.BatchNorm1d(16), nn.GELU())

        self.layer1 = ResidualBlock1D(48, 64, stride=2)   # 240 -> 120
        self.layer2 = ResidualBlock1D(64, 128, stride=2)  # 120 -> 60
        self.layer3 = ResidualBlock1D(128, 192, stride=2) # 60 -> 30
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 2. Scalar Biomarker Branch
        self.bio_mlp = nn.Sequential(
            nn.Linear(num_bio_features, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        # 3. Cross-Feature Fusion Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(192 + 32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_wave: torch.Tensor, x_bio: torch.Tensor) -> torch.Tensor:
        stem = torch.cat([self.b3(x_wave), self.b7(x_wave), self.b15(x_wave)], dim=1)
        w = self.layer1(stem)
        w = self.layer2(w)
        w = self.layer3(w)
        wave_feat = self.global_pool(w).flatten(1)

        bio_feat = self.bio_mlp(x_bio)
        fused = torch.cat([wave_feat, bio_feat], dim=1)
        return self.classifier(fused)

