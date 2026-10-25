"""
MindFit AI - Enterprise Backend Server
FastAPI Service with Native PyTorch CPU Inference, Signal Processing,
Personalized Exercise Recommender, and Explainable AI (SHAP)
"""

import os
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.services.signal_processor import SignalProcessor
from backend.services.inference_engine import InferenceEngine
from backend.services.recommender import ExerciseRecommender
from backend.services.xai_service import XAIService

app = FastAPI(
    title="MindFit AI API",
    description="Deep Learning (1D-CNN) & Recommender Engine for Wearable Anxiety Recognition",
    version="1.0.0"
)

# Enable CORS for local dev & frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
signal_proc = SignalProcessor(target_fs=4.0, window_sec=60)
inference_engine = InferenceEngine()
recommender = ExerciseRecommender()
xai_service = XAIService()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SAMPLES_DIR = os.path.join(BASE_DIR, 'data', 'samples')
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')


class ManualVitalsRequest(BaseModel):
    resting_hr: float = 75.0
    sleep_hours: float = 7.0
    stress_feel: int = 5
    gad7_scores: List[int] = [1, 0, 1, 1, 0, 0, 1]


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "MindFit AI Engine",
        "deep_learning_active": inference_engine.is_deep_learning_active,
        "weights_path": inference_engine.weights_path,
        "classes": inference_engine.CLASSES
    }


@app.get("/api/sample/{profile_id}")
def get_sample_data(profile_id: str):
    """
    1-Click Demo Profiles:
    - 'normal': Apple Watch baseline
    - 'mild': Fitbit transient anxiety
    - 'stress': Empatica E4 acute sympathetic stress
    """
    mapping = {
        "normal": "sample_normal_applewatch.csv",
        "mild": "sample_mild_anxiety_fitbit.csv",
        "stress": "sample_high_stress_empatica.csv"
    }
    filename = mapping.get(profile_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Sample profile not found")

    filepath = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample file missing")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    return {
        "profile_id": profile_id,
        "filename": filename,
        "csv_content": content
    }


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Process wearable time-series CSV, filter noise via Butterworth,
    run 1D-CNN (SE-ResNet1D) inference, generate sports medicine recommendations and SHAP values.
    """
    try:
        content_bytes = await file.read()
        file_text = content_bytes.decode('utf-8', errors='ignore')

        # 1. Parse & filter signal
        bvp_raw, eda_raw, orig_fs = signal_proc.parse_wearable_csv(file_text)
        processed = signal_proc.process_and_window(bvp_raw, eda_raw, orig_fs)

        # 2. 1D-CNN Inference
        prediction = inference_engine.predict_tensor(
            processed["tensor"],
            bvp_mean=processed["bvp_mean"],
            eda_mean=processed["eda_mean"]
        )

        # 3. Exercise Recommendations
        routines = recommender.recommend(
            state=prediction["state"],
            severity_score=prediction["severity_score"]
        )

        # 4. Explainable AI (SHAP Contributions)
        xai_contributions = xai_service.compute_feature_contributions(
            state=prediction["state"],
            bvp_mean=processed["bvp_mean"],
            eda_mean=processed["eda_mean"]
        )

        return {
            "status": "success",
            "filename": file.filename,
            "detected_sampling_rate": round(orig_fs, 1),
            "prediction": prediction,
            "visual_preview": processed["visual_preview"],
            "recommendations": routines[:4],  # Top 4 routines
            "explainability": xai_contributions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file CSV: {str(e)}")


@app.post("/api/predict/manual")
def predict_manual(payload: ManualVitalsRequest):
    """
    Process clinical GAD-7 scale and user-reported physiological vitals.
    """
    try:
        # 1. Clinical Scoring & Prediction
        prediction = inference_engine.predict_manual_vitals(
            resting_hr=payload.resting_hr,
            sleep_hours=payload.sleep_hours,
            stress_feel=payload.stress_feel,
            gad7_scores=payload.gad7_scores
        )

        # 2. Exercise Recommendations
        routines = recommender.recommend(
            state=prediction["state"],
            severity_score=prediction["severity_score"],
            vitals={"sleep_hours": payload.sleep_hours, "gad7_total": prediction.get("gad7_total")}
        )

        # 3. Explainable AI
        xai_contributions = xai_service.compute_feature_contributions(
            state=prediction["state"],
            bvp_mean=payload.resting_hr,
            eda_mean=2.2 if prediction["state"] == "HIGH STRESS" else (1.2 if prediction["state"] == "MILD ANXIETY" else 0.4),
            vitals={"sleep_hours": payload.sleep_hours, "gad7_total": prediction.get("gad7_total")}
        )

        # Generate a simulated visual preview for waveform display
        t = [f"{i*0.6:.1f}s" for i in range(100)]
        base_hr = payload.resting_hr
        bvp_sim = [round(base_hr + 2.5 * (1 if i % 4 == 0 else -0.5), 1) for i in range(100)]
        eda_sim = [round(2.0 if prediction["state"] == "HIGH STRESS" else 0.4, 2) for _ in range(100)]

        return {
            "status": "success",
            "prediction": prediction,
            "visual_preview": {
                "raw_wave": bvp_sim,
                "filtered_bvp": bvp_sim,
                "filtered_eda": eda_sim,
                "timestamps": t
            },
            "recommendations": routines[:4],
            "explainability": xai_contributions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tính toán: {str(e)}")


# Mount frontend static directory if exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
