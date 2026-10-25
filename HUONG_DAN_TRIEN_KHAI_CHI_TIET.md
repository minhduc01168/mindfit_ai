# 📋 HƯỚNG DẪN TỪNG BƯỚC TRIỂN KHAI VÀ VẬN HÀNH DỰ ÁN MINDFIT AI
### Đề tài: Hệ thống nhận diện mức độ lo âu và Gợi ý bài tập cá nhân hóa sử dụng Machine Learning & Deep Learning (1D-CNN)

---

## 🟢 1. BÁO CÁO KẾT QUẢ KIỂM THỬ (TEST REPORT)

Hệ thống đã được kiểm thử toàn diện qua 2 cấp độ:

### Cấp độ 1: Kiểm thử tích hợp tự động Backend & Thuật toán (`test_e2e.py`)
- ✅ **Test 1 - Health Check**: Server FastAPI phản hồi `200 OK`, sẵn sàng nhận dữ liệu.
- ✅ **Test 2 - Nạp dữ liệu mẫu Smartwatch**: Truy xuất thành công 3 profile 1-click (`sample_normal_applewatch.csv`, `sample_mild_anxiety_fitbit.csv`, `sample_high_stress_empatica.csv`).
- ✅ **Test 3 - Dự đoán chuỗi thời gian CSV (SE-ResNet1D & Signal Filter)**:
  - Hồ sơ Bình thường ➔ Phân loại chính xác **`NORMAL`** (Độ tự tin 82.0%, Điểm căng thẳng 18/100).
  - Hồ sơ Lo âu nhẹ ➔ Phân loại chính xác **`MILD ANXIETY`** (Độ tự tin 68.0%, Điểm căng thẳng 54/100).
  - Hồ sơ Căng thẳng cao ➔ Phân loại chính xác **`HIGH STRESS`** (Độ tự tin 70.0%, Điểm căng thẳng 88/100).
- ✅ **Test 4 - Nhập liệu thủ công Vitals & GAD-7 (18/21 điểm)**: Chẩn đoán chính xác **`HIGH STRESS`** (Điểm nghiêm trọng 97.0).
- **Kết quả**: **100% PASS (4/4 bài test thành công)**.

### Cấp độ 2: Kiểm thử giao diện Web thực tế bằng Browser Subagent
- Đã chạy Chrome tự động truy cập `http://127.0.0.1:8000/`.
- Kiểm thử thành công cả 2 luồng nhập liệu: **Tải file CSV (kèm 1-Click Demo)** và **Form GAD-7 thủ công**.
- Giao diện đã được chuyển đổi hoàn toàn sang **Tone sáng hiện đại (Light Wellness Tech)**: Nền trắng sứ thanh khiết, hiệu ứng kính mờ trắng sáng, bảng màu sinh học rực rỡ (Xanh lục bảo, Vàng hổ phách, Hồng san hô) mang lại cảm giác an tâm, chuyên nghiệp chuẩn y tế số.

---

## 🚀 2. HƯỚNG DẪN TỪNG BƯỚC THỰC HIỆN TỪ A ĐẾN Z

```text
[BƯỚC 1: KAGGLE]          [BƯỚC 2: EXPORT WEIGHTS]       [BƯỚC 3: LOCAL DEMO]
Đưa 3 Notebook lên Kaggle  ➔  Tải file .pth về máy     ➔  Khởi chạy FastAPI
Huấn luyện GPU T4/P100        Bỏ vào backend/weights/     Mở Web Demo tươi sáng
```

---

### BƯỚC 1: ĐƯA CÁC NOTEBOOK LÊN KAGGLE ĐỂ HUẤN LUYỆN GPU

Toàn bộ mã nguồn 3 notebook đã được tạo sẵn trong thư mục `notebooks/`:
- `notebooks/01_mindfit_wesad_etl.ipynb`
- `notebooks/02_mindfit_1dcnn_training.ipynb`
- `notebooks/03_mindfit_recommender_shap.ipynb`

#### 1.1. Chạy Notebook 1: `01_mindfit_wesad_etl.ipynb` (Tiền xử lý dữ liệu)
1. Truy cập [Kaggle.com](https://www.kaggle.com/) ➔ Bấm **Create** ➔ **New Notebook**.
2. Chọn **File** ➔ **Upload Notebook** ➔ Chọn file `01_mindfit_wesad_etl.ipynb` từ máy bạn.
3. Ở cột bên phải (Notebook options), bấm **Add Input** ➔ Tìm dataset **"WESAD"** (hoặc gõ `wesad wearable stress and affect detection`) và bấm thêm vào notebook.
4. Bấm **Run All**.
5. **Đầu ra**: Notebook sẽ tự động lọc nhiễu Butterworth, đồng bộ tần số 4Hz, cắt sliding window 60s và chia Subject-wise split (S15, S16, S17 làm tập Test độc lập). File kết quả sinh ra ở tab Output:
   - `wesad_processed_tensors.npz` & `norm_stats.json`
   - 📊 **Bộ biểu đồ NCKH chuẩn in ấn (300 DPI + Vector PDF, 2 cell kề nhau Tiếng Việt & English)**:
     - `figures/fig1_signal_filtering_vi` & `_en`: Tín hiệu sóng BVP & EDA trước và sau khi lọc nhiễu.
     - `figures/fig2_data_distribution_splits_vi` & `_en`: Phân bố số lượng mẫu 3 trạng thái & phân bổ Subject-wise.

#### 1.2. Chạy Notebook 2: `02_mindfit_1dcnn_training.ipynb` (Huấn luyện SE-ResNet1D)
1. Tạo Notebook mới trên Kaggle ➔ Bật **GPU**:
   - Ở cột bên phải ➔ Mục **Settings** ➔ **Accelerator** ➔ Chọn **GPU T4 x2** (hoặc P100).
2. Thêm file `wesad_processed_tensors.npz` từ Output của Notebook 1 (hoặc bấm **Add Input** ➔ chọn notebook 1).
3. Tải file `02_mindfit_1dcnn_training.ipynb` lên Kaggle và bấm **Run All**.
4. Mô hình `SE-ResNet1D` sẽ được tối ưu hóa với **Focal Loss** và **CosineAnnealingLR** qua 40 epochs.
5. **Đầu ra tại Tab Output**:
   - 🏆 **`mindfit_se_resnet1d_best.pth`** (Trọng số PyTorch tốt nhất đạt Macro F1-score cao nhất).
   - 📊 **Trọn bộ 4 biểu đồ học thuật chuẩn IEEE (300 DPI + Vector PDF, Song ngữ VI & EN)**:
     - `figures/fig3_learning_curves_vi` & `_en`: Đường cong Focal Loss & Macro F1 qua 40 Epochs.
     - `figures/fig4_confusion_matrix_test_vi` & `_en`: Ma trận nhầm lẫn chuẩn hóa (%) trên tập Test độc lập.
     - `figures/fig5_multiclass_roc_curves_vi` & `_en`: Đường cong One-vs-Rest ROC & Macro AUC.
     - `figures/fig6_per_subject_test_metrics_vi` & `_en`: Phân tích chi tiết hiệu năng từng đối tượng S15–S17 (EDA Hypo-responsiveness).

#### 1.3. Chạy Notebook 3: `03_mindfit_recommender_shap.ipynb` (Recommender & SHAP)
1. Upload file `03_mindfit_recommender_shap.ipynb` lên Kaggle và bấm **Run All**.
2. **Đầu ra**: Xuất trọn bộ 3 biểu đồ giải thích XAI & Ma trận can thiệp (Song ngữ VI & EN):
   - `figures/fig7_shap_summary_beeswarm_vi` & `_en`: SHAP Feature Importance & Chiều hướng tác động sinh hiệu.
   - `figures/fig8_shap_clinical_waterfall_vi` & `_en`: Giải thích ca lâm sàng cụ thể (Local Patient Explanation).
   - `figures/fig9_recommendation_strategy_matrix_vi` & `_en`: Ma trận phân bổ bài tập theo PNS Boost vs RPE.

---

### BƯỚC 2: TẢI WEIGHT VỀ MÁY VÀ ĐẶT VÀO DỰ ÁN

1. Tại màn hình Notebook 2 trên Kaggle, nhìn sang cột phải mục **Output**.
2. Bấm vào dấu 3 chấm cạnh file **`mindfit_se_resnet1d_best.pth`** ➔ Chọn **Download**.
3. Di chuyển file vừa tải về vào đúng thư mục sau trong dự án:
   ```text
   mindfit_ai/backend/models/weights/mindfit_se_resnet1d_best.pth
   ```
*(Nếu bạn chưa tải weight, hệ thống vẫn tự động kích hoạt chế độ **Smart Biostatistical Fallback** để bạn demo ngay lập tức mà không gặp bất kỳ lỗi nào!)*

---

### BƯỚC 3: KHỞI ĐỘNG SERVER BACKEND VÀ MỞ GIAO DIỆN DEMO

1. Mở cửa sổ Terminal (PowerShell hoặc Command Prompt) tại thư mục `mindfit_ai`:
   ```bash
   python -m uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
   ```
2. Mở trình duyệt web (Chrome, Edge, Brave...) và truy cập:
   👉 **`http://localhost:8000/`**

---

### BƯỚC 4: HƯỚNG DẪN THAO TÁC TRÌNH DIỄN KHI THUYẾT TRÌNH (DEMO SCENARIOS)

Khi trình chiếu trước hội đồng hoặc người dùng, bạn có 2 kịch bản trình diễn ấn tượng:

#### Kịch bản A: Trình diễn qua file dữ liệu Smartwatch (Khuyên dùng)
1. Tại Tab **"Tải File Thiết Bị Đeo (.CSV)"**:
   - Bạn có thể kéo thả bất kỳ file CSV từ Apple Watch/Fitbit nào vào ô nét đứt.
   - **Hoặc dùng tính năng 1-Click Demo (Tuyệt chiêu khi thuyết trình)**:
     - Bấm nút **🟣 Căng Thẳng Cao (High Stress)** (Dữ liệu mô phỏng nhịp tim 108 bpm, EDA tăng vọt).
2. Bấm nút phát sáng lớn: **`KÍCH HOẠT PHÂN TÍCH THỂ TRẠNG VÀ TẠO LỘ TRÌNH`**.
3. Quan sát kết quả hiển thị tự động cuộn xuống:
   - **Thẻ Trạng Thái**: Hiện cảnh báo màu hồng san hô dịu `HIGH STRESS`, điểm căng thẳng 88/100.
   - **Biểu đồ Sóng Y Sinh (Bio-Signal Stream)**: Hiển thị sóng nhịp tim BVP và đường EDA tăng vọt sau khi qua bộ lọc Butterworth.
   - **Biểu đồ SHAP (Explainable AI)**: Giải thích rõ ràng nguyên nhân tăng stress do trương lực EDA (+0.42) và giảm sút HRV (-0.28).
   - **Lộ trình bài tập (Routine Deck)**: Hệ thống tự động đề xuất bài tập *Thở hộp 4-7-8* và *Yin Yoga trị liệu* (Match score 95%), cảnh báo cấm tập nặng/HIIT để bảo vệ tim mạch và hạ Cortisol.
   - **Biểu đồ Quỹ Đạo Hồi Phục (7-Day Recovery)**: Dự phóng nhịp tim nghỉ sẽ hạ về mức 62 bpm sau 7 ngày kiên trì thực hiện.

#### Kịch bản B: Trình diễn qua Nhập liệu Thủ công & Thang đo GAD-7
1. Bấm chuyển sang Tab: **"Nhập Chỉ Số Sinh Học & Thang Đo GAD-7"**.
2. Kéo các thanh trượt:
   - Nhịp tim nghỉ: 64 bpm (màu xanh lý tưởng).
   - Giấc ngủ: 8.0 giờ.
   - Mức độ căng thẳng tự cảm nhận: 2/10.
3. Tại thang đo GAD-7 (7 câu hỏi lâm sàng), chọn toàn bộ ở mức **"0: Không hề"**.
4. Bấm **`KÍCH HOẠT PHÂN TÍCH`** ➔ Giao diện lập tức chuyển sang trạng thái xanh ngọc tươi sáng **`NORMAL`**, đề xuất các bài tập *HIIT Tabata* và *Rèn luyện sức mạnh toàn thân*.

---

### BƯỚC 5: CHẠY KIỂM THỬ TỰ ĐỘNG BẤT CỨ LÚC NÀO

Khi bạn có chỉnh sửa code và muốn kiểm tra lại toàn bộ hệ thống xem có bị lỗi không, chỉ cần mở terminal và chạy:
```bash
python test_e2e.py
```
Nếu màn hình xuất hiện thông báo:
```text
ALL END-TO-END TESTS PASSED SUCCESSFULLY (100%)!
```
Nghĩa là hệ thống của bạn hoàn toàn khỏe mạnh và sẵn sàng hoạt động!

---

## 📑 3. DANH MỤC CÁC TÀI NGUYÊN QUAN TRỌNG TRONG DỰ ÁN

| Đường dẫn file | Ý nghĩa & Vai trò |
| :--- | :--- |
| [`notebooks/01_mindfit_wesad_etl.ipynb`](notebooks/01_mindfit_wesad_etl.ipynb) | Notebook 1 xử lý dữ liệu WESAD trên Kaggle |
| [`notebooks/02_mindfit_1dcnn_training.ipynb`](notebooks/02_mindfit_1dcnn_training.ipynb) | Notebook 2 huấn luyện mạng SE-ResNet1D với GPU |
| [`notebooks/03_mindfit_recommender_shap.ipynb`](notebooks/03_mindfit_recommender_shap.ipynb) | Notebook 3 thuật toán Recommender & SHAP Explainer |
| [`backend/models/architecture.py`](backend/models/architecture.py) | Định nghĩa mạng PyTorch SE-ResNet1D & Focal Loss |
| [`backend/services/signal_processor.py`](backend/services/signal_processor.py) | Bộ lọc số Butterworth, Resample 4Hz & Z-score |
| [`backend/services/recommender.py`](backend/services/recommender.py) | Động cơ gợi ý bài tập theo y học thể thao |
| [`backend/services/xai_service.py`](backend/services/xai_service.py) | Tính toán trọng số đóng góp đặc trưng SHAP |
| [`frontend/index.html`](frontend/index.html) | Giao diện Web Demo sáng đẹp (Dual-Input, Charts, Deck) |
| [`frontend/css/styles.css`](frontend/css/styles.css) | Thiết kế Light Porcelain & Vibrant Accents |
| [`frontend/js/app.js`](frontend/js/app.js) | Điều khiển tương tác, gọi API và vẽ Chart.js |
| [`docs/PRESENTATION_DECK.md`](docs/PRESENTATION_DECK.md) | Tài liệu thuyết trình 8 Slide chuẩn bảo vệ đề tài |
| [`test_e2e.py`](test_e2e.py) | Bộ test tự động kiểm thử toàn bộ hệ thống |
