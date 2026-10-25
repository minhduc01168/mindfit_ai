"""
MindFit AI - Automated End-to-End Integration Test Suite
Validates backend API, Signal Processing, 1D-CNN Inference,
Personalized Recommender, and Explainable AI (SHAP) endpoints.
"""

import sys
import os
import requests
import json
import time

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    print("\n[Test 1] Health Check Endpoint...")
    res = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert res.status_code == 200, f"Health check failed with {res.status_code}"
    data = res.json()
    print(f"   Status: {data['status']}")
    print(f"   Deep Learning Active: {data['deep_learning_active']}")
    print(f"   Weights Path: {data['weights_path']}")
    print("   [PASS] Health check OK")


def test_sample_profiles():
    print("\n[Test 2] Sample CSV Profiles Retrieval...")
    for pid in ['normal', 'mild', 'stress']:
        res = requests.get(f"{BASE_URL}/api/sample/{pid}", timeout=5)
        assert res.status_code == 200, f"Sample profile {pid} failed"
        data = res.json()
        assert len(data['csv_content']) > 100
        print(f"   Profile '{pid}': Loaded {data['filename']} ({len(data['csv_content'])} bytes)")
    print("   [PASS] Sample profiles OK")


def test_predict_csv():
    print("\n[Test 3] CSV Wearable Stream Prediction (Apple Watch / Empatica)...")
    sample_files = [
        ("Normal State", "backend/data/samples/sample_normal_applewatch.csv", "NORMAL"),
        ("Mild Anxiety", "backend/data/samples/sample_mild_anxiety_fitbit.csv", "MILD ANXIETY"),
        ("High Stress", "backend/data/samples/sample_high_stress_empatica.csv", "HIGH STRESS")
    ]

    for label, filepath, expected_state in sample_files:
        assert os.path.exists(filepath), f"File {filepath} missing!"
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f, 'text/csv')}
            res = requests.post(f"{BASE_URL}/api/predict/csv", files=files, timeout=10)

        assert res.status_code == 200, f"Prediction failed: {res.text}"
        payload = res.json()
        pred = payload['prediction']
        print(f"   [{label}] State: {pred['state']} | Confidence: {pred['confidence']*100:.1f}% | Severity: {pred['severity_score']}")
        assert pred['state'] == expected_state, f"Expected {expected_state}, got {pred['state']}"
        assert 'visual_preview' in payload and len(payload['visual_preview']['raw_wave']) == 100
        assert len(payload['recommendations']) > 0
        assert len(payload['explainability']) > 0
    print("   [PASS] CSV prediction OK")


def test_predict_manual():
    print("\n[Test 4] Manual Vitals & GAD-7 Scale Prediction...")
    # High Stress GAD-7 Form
    payload = {
        "resting_hr": 105.0,
        "sleep_hours": 4.5,
        "stress_feel": 9,
        "gad7_scores": [3, 2, 3, 3, 2, 2, 3]  # 18/21 points
    }
    res = requests.post(f"{BASE_URL}/api/predict/manual", json=payload, timeout=5)
    assert res.status_code == 200, f"Manual prediction failed: {res.text}"
    data = res.json()
    pred = data['prediction']
    print(f"   [Manual Stress Form] State: {pred['state']} | GAD-7 Total: {pred['gad7_total']}/21 | Severity: {pred['severity_score']}")
    assert pred['state'] == "HIGH STRESS"
    print("   [PASS] Manual prediction OK")


if __name__ == "__main__":
    print("=" * 60)
    print("MINDFIT AI - END-TO-END AUTOMATED TEST RUNNER")
    print("=" * 60)

    try:
        test_health()
        test_sample_profiles()
        test_predict_csv()
        test_predict_manual()
        print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY (100%)!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
