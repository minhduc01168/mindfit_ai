/**
 * MindFit AI - Client Application Logic
 * Integrates Chart.js, Dual-Input Handling, FastAPI REST Communication,
 * Real-time Waveform Rendering, SHAP Explanations, and Routine Deck.
 */

// Global State
let currentTab = 'upload';
let selectedFile = null;
let currentSampleCsv = null;

// GAD-7 7-Item Clinical Assessment Questions
const GAD7_QUESTIONS = [
  "1. Cảm thấy bồn chồn, lo lắng hoặc không thể ngồi yên?",
  "2. Không thể ngừng lo lắng hoặc không kiểm soát được nỗi lo?",
  "3. Lo lắng quá nhiều về những việc khác nhau?",
  "4. Khó khăn trong việc thư giãn, thả lỏng cơ bắp?",
  "5. Đứng ngồi không yên đến mức khó có thể ngồi yên một chỗ?",
  "6. Trở nên dễ cáu gắt hoặc bực mình một cách vô cớ?",
  "7. Cảm thấy sợ hãi như thể có điều gì tồi tệ sắp xảy ra?"
];
let gad7Scores = [0, 0, 0, 0, 0, 0, 0];

// Chart.js Instances
let waveformChartInstance = null;
let shapChartInstance = null;
let recoveryChartInstance = null;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  initGad7Questions();
  // Auto-load normal preset by default so demo is immediately testable
  loadSamplePreset('normal');
});

// Check Backend Health
async function checkBackendHealth() {
  const pill = document.getElementById('backendStatusPill');
  const text = document.getElementById('backendStatusText');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      text.innerText = data.deep_learning_active 
        ? "AI Engine Sẵn Sàng (Kaggle Weights Active)" 
        : "AI Engine Sẵn Sàng (Biostat Fallback Active)";
      pill.style.borderColor = "rgba(16, 185, 129, 0.4)";
    }
  } catch (err) {
    text.innerText = "Chế độ Standalone Demo";
    pill.style.borderColor = "rgba(245, 158, 11, 0.4)";
  }
}

// Switch Input Tab
function switchInputTab(tab) {
  currentTab = tab;
  const uploadBtn = document.getElementById('tabUploadBtn');
  const manualBtn = document.getElementById('tabManualBtn');
  const uploadContent = document.getElementById('tabUploadContent');
  const manualContent = document.getElementById('tabManualContent');

  if (tab === 'upload') {
    uploadBtn.classList.add('active');
    manualBtn.classList.remove('active');
    uploadContent.style.display = 'block';
    manualContent.style.display = 'none';
  } else {
    manualBtn.classList.add('active');
    uploadBtn.classList.remove('active');
    manualContent.style.display = 'block';
    uploadContent.style.display = 'none';
  }
}

// Drag & Drop Handlers
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.add('dragover');
}
function handleDragLeave(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('dragover');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    processSelectedFile(e.dataTransfer.files[0]);
  }
}
function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    processSelectedFile(e.target.files[0]);
  }
}
function processSelectedFile(file) {
  selectedFile = file;
  currentSampleCsv = null;
  const info = document.getElementById('selectedFileInfo');
  const label = document.getElementById('fileNameLabel');
  info.style.display = 'inline-flex';
  label.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
}

// 1-Click Demo Profiles (Presentation Helper)
async function loadSamplePreset(profileId) {
  try {
    const res = await fetch(`/api/sample/${profileId}`);
    const data = await res.json();
    currentSampleCsv = data.csv_content;
    selectedFile = null;

    const info = document.getElementById('selectedFileInfo');
    const label = document.getElementById('fileNameLabel');
    info.style.display = 'inline-flex';
    label.innerText = `Đã nạp hồ sơ demo: ${data.filename}`;
    
    // Auto-scroll slightly or highlight
    info.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error("Could not load preset:", err);
  }
}

// Sliders Live Update
function updateSliderVal(id, suffix) {
  const input = document.getElementById(`input${id}`);
  const valDisplay = document.getElementById(`val${id}`);
  valDisplay.innerText = `${input.value}${suffix}`;
}

// Initialize GAD-7 Scale UI
function initGad7Questions() {
  const container = document.getElementById('gad7QuestionsList');
  container.innerHTML = '';

  GAD7_QUESTIONS.forEach((qText, qIdx) => {
    const item = document.createElement('div');
    item.className = 'gad7-item';
    item.innerHTML = `
      <div class="gad7-question-text">${qText}</div>
      <div class="gad7-pills-row">
        <button type="button" class="gad7-pill-btn selected" onclick="selectGad7Score(${qIdx}, 0, this)">0: Không hề</button>
        <button type="button" class="gad7-pill-btn" onclick="selectGad7Score(${qIdx}, 1, this)">1: Vài ngày</button>
        <button type="button" class="gad7-pill-btn" onclick="selectGad7Score(${qIdx}, 2, this)">2: > 7 ngày</button>
        <button type="button" class="gad7-pill-btn" onclick="selectGad7Score(${qIdx}, 3, this)">3: Mỗi ngày</button>
      </div>
    `;
    container.appendChild(item);
  });
  updateGad7Total();
}

function selectGad7Score(qIdx, score, btnElem) {
  gad7Scores[qIdx] = score;
  const parent = btnElem.parentElement;
  parent.querySelectorAll('.gad7-pill-btn').forEach(b => b.classList.remove('selected'));
  btnElem.classList.add('selected');
  updateGad7Total();
}

function updateGad7Total() {
  const total = gad7Scores.reduce((a, b) => a + b, 0);
  document.getElementById('gad7TotalScore').innerText = total;
  const tag = document.getElementById('gad7SeverityTag');

  if (total <= 4) {
    tag.innerText = "Bình thường (Minimal)";
    tag.style.background = "rgba(16, 185, 129, 0.15)";
    tag.style.color = "var(--color-normal)";
  } else if (total <= 9) {
    tag.innerText = "Lo âu nhẹ (Mild)";
    tag.style.background = "rgba(245, 158, 11, 0.15)";
    tag.style.color = "var(--color-mild)";
  } else if (total <= 14) {
    tag.innerText = "Lo âu vừa (Moderate)";
    tag.style.background = "rgba(249, 115, 22, 0.15)";
    tag.style.color = "#f97316";
  } else {
    tag.innerText = "Căng thẳng nặng (Severe)";
    tag.style.background = "rgba(236, 72, 153, 0.15)";
    tag.style.color = "var(--color-stress)";
  }
}

// Run Analysis Trigger
async function runAnalysis() {
  const loading = document.getElementById('loadingState');
  const results = document.getElementById('resultsSection');
  const analyzeBtn = document.getElementById('analyzeBtn');

  analyzeBtn.disabled = true;
  loading.style.display = 'block';
  results.style.display = 'none';

  loading.scrollIntoView({ behavior: 'smooth', block: 'center' });

  try {
    let payload;
    if (currentTab === 'upload') {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else if (currentSampleCsv) {
        const blob = new Blob([currentSampleCsv], { type: 'text/csv' });
        formData.append('file', blob, 'sample_stream.csv');
      } else {
        alert("Vui lòng chọn file CSV hoặc chọn 1 Hồ sơ thử nghiệm phía trên!");
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        return;
      }

      const res = await fetch('/api/predict/csv', {
        method: 'POST',
        body: formData
      });
      payload = await res.json();
    } else {
      // Manual Vitals
      const manualData = {
        resting_hr: parseFloat(document.getElementById('inputRestingHR').value),
        sleep_hours: parseFloat(document.getElementById('inputSleepHours').value),
        stress_feel: parseInt(document.getElementById('inputStressFeel').value),
        gad7_scores: gad7Scores
      };

      const res = await fetch('/api/predict/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(manualData)
      });
      payload = await res.json();
    }

    // Render Payload Results
    setTimeout(() => {
      loading.style.display = 'none';
      results.style.display = 'block';
      analyzeBtn.disabled = false;
      renderAllResults(payload);
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 700);

  } catch (err) {
    console.error("Error during analysis:", err);
    alert("Có lỗi xảy ra trong quá trình tính toán: " + err.message);
    loading.style.display = 'none';
    analyzeBtn.disabled = false;
  }
}

// Master Render Function
function renderAllResults(data) {
  const pred = data.prediction;
  renderStateBanner(pred);
  renderWaveformChart(data.visual_preview);
  renderShapExplainer(data.explainability);
  renderRoutines(data.recommendations);
  renderRecoveryTrend(pred.state, pred.severity_score);
  lucide.createIcons();
}

// 1. Render State Classification Banner
function renderStateBanner(pred) {
  const banner = document.getElementById('stateBanner');
  const icon = document.getElementById('stateIcon');
  const badge = document.getElementById('stateBadgeTitle');
  const confPill = document.getElementById('confidencePill');
  const modelPill = document.getElementById('modelSourcePill');
  const heading = document.getElementById('stateHeading');
  const desc = document.getElementById('stateDesc');
  const sevScore = document.getElementById('severityScore');
  const barFill = document.getElementById('meterBarFill');

  banner.className = 'state-banner glass-panel';
  sevScore.innerText = pred.severity_score;
  barFill.style.width = `${Math.min(100, pred.severity_score)}%`;
  confPill.innerText = `Độ tự tin: ${(pred.confidence * 100).toFixed(1)}%`;
  modelPill.innerText = pred.model_source || "PyTorch SE-ResNet1D";

  if (pred.state === 'HIGH STRESS') {
    banner.classList.add('stress');
    icon.innerText = '🟣';
    badge.innerText = 'HIGH STRESS (CĂNG THẲNG CAO)';
    heading.innerText = 'Cảnh Báo: Trạng Thái Thần Kinh Căng Thẳng & Cortisol Tăng Vọt';
    desc.innerText = 'Tín hiệu ghi nhận phản xạ giao cảm cấp tính (Sympathetic Arousal) với độ dẫn điện da EDA tăng cao và suy giảm nghiêm trọng độ biến thiên nhịp tim (HRV). Khuyến nghị dừng ngay các bài tập tạ hoặc HIIT gắng sức để tránh quá tải trục thượng thận HPA.';
  } else if (pred.state === 'MILD ANXIETY') {
    banner.classList.add('mild');
    icon.innerText = '🟡';
    badge.innerText = 'MILD ANXIETY (LO ÂU NHẸ)';
    heading.innerText = 'Trạng Thái: Bồn Chồn & Dao Động Tâm Lý';
    desc.innerText = 'Hệ thần kinh xuất hiện các xung nhạy cảm, nhịp tim nghỉ có xu hướng dao động thất thường. Khuyến nghị áp dụng các bài tập nhịp điệu Zone 2, Pilates hoặc Vinyasa Yoga để kích thích tuần hoàn endorphin và giải tỏa áp lực nhẹ nhàng.';
  } else {
    banner.classList.add('normal');
    icon.innerText = '🟢';
    badge.innerText = 'NORMAL (THỂ TRẠNG BÌNH THƯỜNG)';
    heading.innerText = 'Trạng Thái: Thư Thái & Cân Bằng Sinh Học Tuyệt Vời';
    desc.innerText = 'Chỉ số HRV dồi dào và nhịp tim nghỉ ổn định chứng minh hệ thần kinh đối giao cảm (PNS) đang phục hồi hoàn hảo. Bạn hoàn toàn có thể tự tin tham gia các bài tập rèn luyện thể lực, Cardio cường độ cao hoặc nâng tạ.';
  }
}

// 2. Render Bio-Signal Waveform Chart (Chart.js)
function renderWaveformChart(preview) {
  const ctx = document.getElementById('waveformChart').getContext('2d');
  if (waveformChartInstance) {
    waveformChartInstance.destroy();
  }

  const labels = preview.timestamps;
  waveformChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'BVP Pulse (Nhịp tim đã lọc Butterworth)',
          data: preview.filtered_bvp,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          yAxisID: 'y1'
        },
        {
          label: 'EDA Conductance (Điện trở da µS)',
          data: preview.filtered_eda,
          borderColor: '#c084fc',
          backgroundColor: 'rgba(192, 132, 252, 0.08)',
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.2,
          yAxisID: 'y2'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          padding: 10,
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1'
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#64748b', maxTicksLimit: 12 }
        },
        y1: {
          type: 'linear',
          position: 'left',
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#0284c7', font: { weight: 'bold' } },
          title: { display: true, text: 'BVP Pulse (BPM)', color: '#0284c7', font: { weight: 'bold' } }
        },
        y2: {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#9333ea', font: { weight: 'bold' } },
          title: { display: true, text: 'EDA (µS)', color: '#9333ea', font: { weight: 'bold' } }
        }
      }
    }
  });
}

// 3. Render Explainable AI (SHAP Waterfall / Bars)
function renderShapExplainer(contributions) {
  const ctx = document.getElementById('shapChart').getContext('2d');
  if (shapChartInstance) {
    shapChartInstance.destroy();
  }

  const labels = contributions.map(c => c.code);
  const dataVals = contributions.map(c => c.shap_value);
  const colors = dataVals.map(v => v >= 0 ? '#e11d48' : '#059669');

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'SHAP Value (+ Tăng Stress / - Giảm Stress)',
        data: dataVals,
        backgroundColor: colors,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          padding: 10,
          callbacks: {
            title: (items) => contributions[items[0].dataIndex].feature,
            label: (item) => `Độ tác động SHAP: ${item.raw > 0 ? '+' : ''}${item.raw}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#64748b' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#0f172a', font: { weight: 'bold' } }
        }
      }
    }
  });

  // Render Insight Cards
  const list = document.getElementById('shapInsightsList');
  list.innerHTML = '';
  contributions.forEach(c => {
    const isPos = c.shap_value >= 0;
    const card = document.createElement('div');
    card.className = 'shap-insight-item';
    card.innerHTML = `
      <span class="shap-impact-badge ${isPos ? 'impact-pos' : 'impact-neg'}">
        ${isPos ? '+' : ''}${c.shap_value}
      </span>
      <div class="insight-content">
        <strong>${c.feature} (${c.measured})</strong>
        <p>${c.clinical_insight}</p>
      </div>
    `;
    list.appendChild(card);
  });
}

// 4. Render Recommended Sports Medicine Routines
function renderRoutines(routines) {
  const grid = document.getElementById('routinesGrid');
  grid.innerHTML = '';
  document.getElementById('routineCountBadge').innerText = `${routines.length} Bài tập phù hợp nhất`;

  routines.forEach(r => {
    const card = document.createElement('div');
    card.className = 'routine-card';
    card.innerHTML = `
      <div class="routine-top-row">
        <span class="routine-icon">${r.icon || '🏃'}</span>
        <span class="match-score-pill">Phù hợp: ${r.match_score}%</span>
      </div>
      <h4 class="routine-name">${r.name}</h4>
      <div class="routine-meta-pills">
        <span class="meta-tag">⏱️ ${r.duration_min} phút</span>
        <span class="meta-tag">⚡ Cường độ: ${r.intensity}</span>
        <span class="meta-tag">🧠 ${r.nervous_system}</span>
      </div>
      <p class="routine-desc">${r.description}</p>
      <div class="routine-rationale">
        <strong>Lý do chỉ định:</strong> ${r.rationale}
      </div>
      <div class="routine-steps-toggle">
        <details class="steps-details">
          <summary>Xem hướng dẫn từng bước (${r.steps.length} bước)</summary>
          <ol class="steps-list">
            ${r.steps.map(s => `<li>${s}</li>`).join('')}
          </ol>
        </details>
      </div>
    `;
    grid.appendChild(card);
  });
}

// 5. Render 7-Day Recovery Simulation Trend
function renderRecoveryTrend(state, currentSeverity) {
  const ctx = document.getElementById('recoveryChart').getContext('2d');
  if (recoveryChartInstance) {
    recoveryChartInstance.destroy();
  }

  const days = ['Hôm nay', 'Ngày 2', 'Ngày 3', 'Ngày 4', 'Ngày 5', 'Ngày 6', 'Ngày 7'];
  let hrTrend = [];
  let anxietyTrend = [];

  const baseSev = Math.max(15, currentSeverity);
  const baseHr = state === 'HIGH STRESS' ? 104 : (state === 'MILD ANXIETY' ? 84 : 66);

  for (let i = 0; i < 7; i++) {
    const factor = Math.exp(-0.25 * i);
    hrTrend.push(Math.round(62 + (baseHr - 62) * factor));
    anxietyTrend.push(Math.round(15 + (baseSev - 15) * factor));
  }

  recoveryChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: days,
      datasets: [
        {
          label: 'Nhịp Tim Nghỉ Dự Báo (Resting HR bpm)',
          data: hrTrend,
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.08)',
          borderWidth: 2,
          tension: 0.3,
          pointBackgroundColor: '#0284c7',
          yAxisID: 'yHr'
        },
        {
          label: 'Chỉ Số Lo Âu Dự Báo (Anxiety / Stress Score)',
          data: anxietyTrend,
          borderColor: '#059669',
          backgroundColor: 'rgba(5, 150, 105, 0.08)',
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.3,
          pointBackgroundColor: '#059669',
          yAxisID: 'ySev'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#334155', font: { weight: '600' } }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#64748b' }
        },
        yHr: {
          type: 'linear',
          position: 'left',
          grid: { color: 'rgba(0, 0, 0, 0.05)' },
          ticks: { color: '#0284c7', font: { weight: 'bold' } },
          title: { display: true, text: 'Nhịp tim (bpm)', color: '#0284c7', font: { weight: 'bold' } }
        },
        ySev: {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#059669', font: { weight: 'bold' } },
          title: { display: true, text: 'Điểm Lo âu (0 - 100)', color: '#059669', font: { weight: 'bold' } }
        }
      }
    }
  });
}
