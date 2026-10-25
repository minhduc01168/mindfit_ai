import numpy as np
import pandas as pd
import os

target_dir = os.path.join(os.path.dirname(__file__), 'backend', 'data', 'samples')
os.makedirs(target_dir, exist_ok=True)
t = np.linspace(0, 60, 240)  # 4Hz for 60 seconds

# 1. Normal State: HR ~64 bpm, RSA present, EDA low ~0.35 uS
hr_normal = 64.0 + 3.0 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.5, 240)
eda_normal = 0.35 + 0.05 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.01, 240)
df_normal = pd.DataFrame({
    'timestamp_sec': np.round(t, 2),
    'heart_rate': np.round(hr_normal, 2),
    'eda_microsiemens': np.round(eda_normal, 4)
})
df_normal.to_csv(os.path.join(target_dir, 'sample_normal_applewatch.csv'), index=False)

# 2. Mild Anxiety: HR ~85 bpm, higher variability, EDA moderate ~1.8 uS
hr_mild = 85.0 + 5.0 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 1.2, 240)
eda_mild = 1.8 + 0.3 * np.sin(2 * np.pi * 0.08 * t) + np.random.normal(0, 0.03, 240)
df_mild = pd.DataFrame({
    'timestamp_sec': np.round(t, 2),
    'heart_rate': np.round(hr_mild, 2),
    'eda_microsiemens': np.round(eda_mild, 4)
})
df_mild.to_csv(os.path.join(target_dir, 'sample_mild_anxiety_fitbit.csv'), index=False)

# 3. High Stress: HR ~108 bpm, suppressed vagal tone, EDA elevated ~6.2 uS with bursts
hr_stress = 108.0 + 2.0 * np.sin(2 * np.pi * 0.15 * t) + np.random.normal(0, 1.8, 240)
eda_stress = 5.8 + 1.2 * np.exp(-((t - 25)**2) / 80.0) + 0.9 * np.exp(-((t - 45)**2) / 60.0) + np.random.normal(0, 0.05, 240)
df_stress = pd.DataFrame({
    'timestamp_sec': np.round(t, 2),
    'heart_rate': np.round(hr_stress, 2),
    'eda_microsiemens': np.round(eda_stress, 4)
})
df_stress.to_csv(os.path.join(target_dir, 'sample_high_stress_empatica.csv'), index=False)

print(f"Generated samples successfully in {target_dir}")
