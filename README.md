# 🧠 MindFit AI
### Hệ Thống Nhận Diện Mức Độ Lo Âu & Gợi Ý Bài Tập Cá Nhân Hóa Sử Dụng Machine Learning & Deep Learning (1D-CNN)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-GPU_T4-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/)
[![XAI](https://img.shields.io/badge/Explainability-SHAP-blue.svg)](https://shap.readthedocs.io/)

---

## 📖 1. Giới Thiệu Đề Tài
**MindFit AI** là hệ thống y tế số khép kín (*Closed-Loop Wearable AI*) tự động hóa quá trình xây dựng lộ trình tập luyện thể chất cá nhân hóa nhằm giảm thiểu lo âu (anxiety) và cải thiện sức khỏe tâm lý dựa trên dữ liệu sinh lý học từ thiết bị đeo thông minh (Apple Watch, Fitbit, Empatica E4).

- **Deep Learning (1D-CNN)**: Sử dụng kiến trúc đa thang đo kết hợp cơ chế chú ý kênh **SE-ResNet1D (Squeeze-and-Excitation 1D-CNN)** để trích xuất đặc trưng ẩn từ tín hiệu thời gian thực (BVP/PPG và EDA) trên bộ dữ liệu WESAD/PMData.
- **Explainable AI (SHAP)**: Minh bạch hóa các yếu tố quyết định của mô hình AI, giúp giải thích vì sao Yoga phục hồi được ưu tiên thay vì HIIT vào ngày căng thẳng cao.
- **Recommender System**: Gợi ý lộ trình bài tập y học thể thao nhằm hạ hormone Cortisol và kích hoạt hệ thần kinh đối giao cảm (PNS).
- **Dual-Input Modern Web Demo**: Hỗ trợ 2 phương thức nhập liệu: Tải file `.csv` thiết bị đeo (kèm nút 1-click demo) và Nhập liệu tức thời với thang đo lâm sàng chuẩn hóa **GAD-7**.

---

## 📂 2. Cấu Trúc Thư Mục Dự Án

```text
mindfit_ai/
├── notebooks/                                  # Bộ 3 Notebook huấn luyện trên Kaggle
│   ├── 01_mindfit_wesad_etl.ipynb              # ETL: Lọc Butterworth, Resample 4Hz, Windowing 60s
│   ├── 02_mindfit_1dcnn_training.ipynb         # Train GPU: SE-ResNet1D, Focal Loss, Export Weights
│   └── 03_mindfit_recommender_shap.ipynb       # Recommender: Exercise Embeddings & SHAP Explainer
│
├── backend/                                    # Local Inference Server (FastAPI)
│   ├── models/
│   │   ├── architecture.py                     # Định nghĩa PyTorch MindFitSEResNet1D & FocalLoss
│   │   └── weights/                            # Nơi chứa file weight .pth tải từ Kaggle về
│   │       └── mindfit_se_resnet1d_best.pth
│   ├── services/
│   │   ├── signal_processor.py                 # Lọc số Butterworth & Chuẩn hóa Z-score
│   │   ├── inference_engine.py                 # PyTorch CPU Native Inference & Smart Fallback
│   │   ├── recommender.py                      # Content-Based Sports Medicine Engine
│   │   └── xai_service.py                      # Dịch vụ giải thích đóng góp đặc trưng SHAP
│   ├── data/
│   │   ├── exercises/exercises.json            # Cơ sở dữ liệu bài tập chuẩn y khoa
│   │   └── samples/                            # Bộ dữ liệu mẫu CSV từ Smartwatch
│   │       ├── sample_normal_applewatch.csv
│   │       ├── sample_mild_anxiety_fitbit.csv
│   │       └── sample_high_stress_empatica.csv
│   └── server.py                               # FastAPI REST Server & Static File Server
│
├── frontend/                                   # Giao diện Web Dark Glassmorphism Hiện Đại
│   ├── index.html                              # Trang chính: Dual-Input, Waveform, SHAP, Routine Deck
│   ├── css/styles.css                          # Thiết kế Mindful Cyber-Aesthetics
│   └── js/app.js                               # Logic tương tác, Chart.js & REST API client
│
├── docs/
│   └── PRESENTATION_DECK.md                    # Tài liệu thuyết trình 8 Slide chuẩn quốc tế
├── generate_samples.py                         # Script sinh dữ liệu mẫu thời gian thực
├── test_e2e.py                                 # Bộ kiểm thử tích hợp tự động End-to-End
└── README.md
```

---

## 🚀 3. Hướng Dẫn Huấn Luyện Trên Kaggle Tận Dụng GPU

### Bước 1: Tiền xử lý dữ liệu (Notebook 01)
1. Tạo một Notebook mới trên Kaggle, đính kèm dataset **WESAD** (hoặc upload từ máy).
2. Tải file `notebooks/01_mindfit_wesad_etl.ipynb` lên Kaggle và chọn **Run All**.
3. Notebook sẽ lọc tín hiệu, chia Subject-wise split (S15, S16, S17 làm tập Test độc lập chống Leakage) và xuất ra file:
   - `wesad_processed_tensors.npz`
   - `norm_stats.json`

### Bước 2: Huấn luyện SE-ResNet1D (Notebook 02)
1. Tạo Notebook thứ 2 trên Kaggle, bật **GPU T4 x2** (Accelerator -> GPU).
2. Nạp file `wesad_processed_tensors.npz` từ Output của Notebook 01.
3. Tải file `notebooks/02_mindfit_1dcnn_training.ipynb` lên và chọn **Run All**.
4. Mô hình sẽ được tối ưu hóa với **Focal Loss** và **CosineAnnealingLR**.
5. Sau khi hoàn thành, vào tab **Output** của Kaggle, tải file weight PyTorch:
   - `mindfit_se_resnet1d_best.pth`
6. Đặt file này vào thư mục `backend/models/weights/` trong dự án máy cục bộ.

---

## 💻 4. Hướng Dẫn Khởi Chạy Local Web Demo

### Khởi động Server Backend & Web App:
Mở terminal trong thư mục dự án và chạy lệnh:
```bash
python -m uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

Truy cập giao diện Web Demo tại trình duyệt:
👉 **`http://localhost:8000/`**

> **Lưu ý tiện ích**: Ngay cả khi bạn chưa kịp tải weight từ Kaggle về, hệ thống đã tích hợp sẵn cơ chế **Smart Biostatistical Fallback**, cho phép bạn trải nghiệm trọn vẹn 100% tính năng của giao diện (vẽ sóng nhịp tim, biểu đồ SHAP, lộ trình bài tập) mà không gặp bất kỳ lỗi nào!

---

## 🧪 5. Kiểm Thử Tự Động (Automated Testing)

Chạy bộ kiểm thử tích hợp tự động để kiểm tra toàn bộ luồng hoạt động:
```bash
python test_e2e.py
```
Bộ test sẽ tự động xác thực:
- ✅ Endpoint kiểm tra sức khỏe `/api/health`.
- ✅ Truy xuất các hồ sơ mẫu 1-click `/api/sample/{profile}`.
- ✅ Tải file CSV và suy luận sóng nhịp tim, nhãn dự đoán và chỉ số SHAP.
- ✅ Nhập liệu thủ công chỉ số sinh học và thang đo lâm sàng GAD-7.

---

## 📊 6. Tài Liệu Thuyết Trình
Tài liệu thuyết trình chi tiết 8 slide chuẩn học thuật kèm kịch bản thuyết minh được lưu trữ tại:
👉 [`docs/PRESENTATION_DECK.md`](docs/PRESENTATION_DECK.md)
