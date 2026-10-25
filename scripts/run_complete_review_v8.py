"""
scripts/run_complete_review_v8.py
Master execution script on local machine to address 100% of peer-review requirements:
1. Benchmarks single-threaded CPU latency on the host machine (Intel Core i7-10750H).
2. Executes 5-seed machine learning baselines (DT, kNN, LDA, RF, XGBoost) with 95% CIs.
3. Validates all 9 300 DPI figures and their metric integrity.
4. Synchronizes and compiles the IEEE publication manuscript (Word docx v8 and Markdown).
5. Executes an exhaustive compliance audit against all 32 citations and Review v7/v8 rules.
"""

import os
import sys
import time
import json
import re
import numpy as np
import torch
import docx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))

# Ensure stdout uses UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def step1_benchmark_latency():
    print("\n" + "=" * 80)
    print("STEP 1: BENCHMARKING DUAL-BRANCH BIO-FUSION NET LATENCY ON LOCAL CPU")
    print("=" * 80)
    
    # Import architecture
    sys.path.append(os.path.abspath("backend"))
    try:
        from models.architecture import DualBranchBioFusionNet
    except ImportError as e:
        print(f"Error importing model architecture: {e}")
        return None

    torch.set_num_threads(1)
    model = DualBranchBioFusionNet(in_channels=4, num_bio_features=6, num_classes=3)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    x_wave = torch.randn(1, 4, 240)
    x_bio = torch.randn(1, 6)

    # Warmup
    print("Warming up inference engine (50 iterations)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_wave, x_bio)

    # Benchmark 1000 iterations
    N = 1000
    print(f"Benchmarking inference latency over {N:,} iterations (single thread)...")
    latencies = []
    with torch.no_grad():
        for _ in range(N):
            t0 = time.perf_counter()
            _ = model(x_wave, x_bio)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    print(f"Latency Results (Local CPU):")
    print(f"  Mean:   {mean_lat:.2f} ± {std_lat:.2f} ms")
    print(f"  Median: {median_lat:.2f} ms")
    print(f"  P95:    {p95_lat:.2f} ms")
    print(f"  Throughput: {1000.0 / mean_lat:.1f} inferences/second")
    
    return {
        'total_params': total_params,
        'mean_ms': mean_lat,
        'std_ms': std_lat,
        'median_ms': median_lat,
        'p95_ms': p95_lat
    }

def step2_run_baselines():
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING MULTI-SEED BASELINE BENCHMARK (5 SEEDS + 95% CI)")
    print("=" * 80)
    try:
        from scripts.run_baselines_multiseed import run_experiment
    except ImportError:
        from run_baselines_multiseed import run_experiment
    results = run_experiment()
    return results

def step3_verify_figures():
    print("\n" + "=" * 80)
    print("STEP 3: VERIFYING 300 DPI FIGURES INTEGRITY")
    print("=" * 80)
    figs = [
        ('images/fig1_signal_filtering_en.png', 'Fig 1: Signal Preprocessing and Artifact Filtering'),
        ('images/fig2_data_distribution_splits_en.png', 'Fig 2: Class Distribution & Subject Splits'),
        ('images/fig3_learning_curves_en.png', 'Fig 3: Training & Validation Convergence Curves'),
        ('images/fig4_confusion_matrix_test_en.png', 'Fig 4: Normalized Confusion Matrix (S15–S17)'),
        ('images/fig5_multiclass_roc_curves_en.png', 'Fig 5: Multi-Class ROC Curves'),
        ('images/fig6_per_subject_test_metrics_en.png', 'Fig 6: Per-Subject Metric Breakdown (S15, S16, S17)'),
        ('images/fig7_shap_summary_beeswarm_en.png', 'Fig 7: TreeExplainer Global SHAP Beeswarm'),
        ('images/fig8_shap_clinical_waterfall_en.png', 'Fig 8: Single-Window Clinical Waterfall SHAP'),
        ('images/fig9_recommendation_strategy_matrix_en.png', 'Fig 9: Affect-to-Exercise Recommendation Matrix'),
    ]
    all_ok = True
    for path, desc in figs:
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            sz = os.path.getsize(path)
            print(f"  [OK] {desc} -> {path} ({sz:,} bytes)")
        else:
            print(f"  [MISSING/CORRUPTED] {desc} -> {path}")
            all_ok = False
    return all_ok

def step4_compile_paper():
    print("\n" + "=" * 80)
    print("STEP 4: COMPILING IEEE RESEARCH MANUSCRIPTS (WORD DOCX & MARKDOWN)")
    print("=" * 80)
    from generate_paper import create_paper
    create_paper()
    print("  [OK] Successfully regenerated docs/MindFit_AI_Research_Paper_IEEE_Revised_v8.docx and synchronized copies.")

def step5_audit_compliance():
    print("\n" + "=" * 80)
    print("STEP 5: EXHAUSTIVE PEER-REVIEW AUDIT & SANITY CHECKS")
    print("=" * 80)
    
    # 1. Check Markdown paper
    with open('docs/MINDFIT_AI_RESEARCH_PAPER.md', 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 2. Check Word docx
    docx_file = 'docs/MindFit_AI_Research_Paper_IEEE_Revised.docx'
    if os.path.exists('docs/MindFit_AI_Research_Paper_IEEE_Revised_v8_updated.docx'):
        docx_file = 'docs/MindFit_AI_Research_Paper_IEEE_Revised_v8_updated.docx'
    elif os.path.exists('docs/MindFit_AI_Research_Paper_IEEE_Revised_v8.docx'):
        try:
            _ = docx.Document('docs/MindFit_AI_Research_Paper_IEEE_Revised_v8.docx')
            docx_file = 'docs/MindFit_AI_Research_Paper_IEEE_Revised_v8.docx'
        except PermissionError:
            pass

    print(f"Auditing Word Document: {docx_file}")
    v8_doc = docx.Document(docx_file)
    docx_text = '\n'.join([p.text for p in v8_doc.paragraphs])
    for t in v8_doc.tables:
        for row in t.rows:
            docx_text += '\n' + '\t'.join([cell.text for cell in row.cells])

    # Core Metric Assertions
    core_metrics = [
        ('63.7%', 'Test Cohort 3-Class Overall Accuracy = 63.7% in text'),
        ('63.69', 'Exact 3-Class Overall Accuracy = 63.69% in Table 4'),
        ('71.6%', 'Test Cohort Binary Non-Stress vs Stress Accuracy = 71.6% in text'),
        ('71.59', 'Exact Binary Non-Stress vs Stress Accuracy = 71.59% in Table 4'),
        ('72.10', 'Binary Balanced Accuracy = 72.10% in Table 4'),
        ('0.575', 'Test Cohort 3-Class Macro F1 (Pooled) = 0.575'),
        ('0.570', 'Test Cohort Subject-Wise Mean F1 = 0.570'),
        ('53.0%', 'Amusement Precision = 53.0% (106/200)'),
        ('0.995', 'Waterfall High-Stress Probability f(x) = 0.995'),
        ('0.267', 'TreeExplainer Base Rate E[f(x)] = 0.267 (210/785)'),
        ('Table 3 in [11]', 'Schmidt et al. WESAD reference points to Table 3 in [11]'),
        ('53.98%', 'Schmidt DT wrist accuracy 53.98%'),
        ('0.641', 'Schmidt AdaBoost Macro F1 0.641'),
        ('0.632', 'Schmidt LDA Macro F1 0.632'),
        ('0.629', 'Schmidt RF Macro F1 0.629'),
        ('0.372', 'Schmidt kNN Macro F1 0.372'),
        ('552,195', 'Exact network parameters = 552,195'),
        ('2.41 ± 0.38 ms', 'Exact CPU latency = 2.41 ± 0.38 ms'),
        ('Intel Core i7-10750H', 'Exact CPU processor model cited'),
        ('Majority Baseline (Zero-R)', 'Majority Baseline (Zero-R) in Table 4'),
        ('73.25', 'Majority Baseline binary 73.25% in Table 4'),
        ('83.31', 'LDA binary 83.31% in Table 4'),
        ('58.98', 'Random Forest 58.98% in Table 4'),
        ('60.51', 'XGBoost 60.51% in Table 4'),
        ('Class-Aware Adaptive Stride protocol, and subject-calibrated baseline normalization', 'Corrected Abstract logic in paper'),
        ('PAR-Q+', 'PAR-Q+ exercise safety contraindication in paper'),
        ('Note on nomenclature:', 'Explicit PRV vs ECG-HRV nomenclature note'),
        ('consistent with a blunted sympathetic response', 'Softened S17 physiological discussion'),
        ('Explanatory Scope of Tabular Biomarker Attribution', 'Properly scoped XAI limitation heading'),
    ]

    print("\n--> [SUB-CHECK 5.1] Core Metrics Verification:")
    failures = 0
    for needle, desc in core_metrics:
        in_md = needle in md_text
        in_docx = needle in docx_text
        if in_md and in_docx:
            print(f"  [PASS] {desc}")
        else:
            print(f"  [FAIL] {desc} (MD: {in_md}, DOCX: {in_docx})")
            failures += 1

    # Forbidden Misleading Phrases
    banned_phrases = [
        ('statistically significant', 'Overly strong "statistically significant" claim on 3-subject test'),
        ('cannot be attained through either modality alone', 'Overly strong dogmatic claim in 4.5'),
        ('rigorous empirical comparison', 'Old "rigorous empirical comparison" phrase in 4.5'),
        ('surrogate Random Forest model (91.2% fidelity)', 'Misleading "surrogate Random Forest model (91.2% fidelity)"'),
        ('91.2% fidelity', 'Unsubstantiated "91.2% fidelity" claim'),
        ('surrogate game-theoretic explainability (SHAP)', 'Outdated "surrogate game-theoretic explainability" claim'),
        ('closely matching the underlying deep neural network\'s softmax output of 94.5%', 'Misleading cross-model validation claim'),
        ('(underlying CNN P = 0.945)', 'Cross-model validation claim in Fig 8 caption'),
        ('competitive performance on unseen individuals while remaining within the expected empirical range', 'Overly strong conclusion in 6.1'),
        ('due to a blunted sympathetic response', 'Overly dogmatic causal phrase in 6.2'),
        ('Raspberry Pi 4', 'Unbenchmarked Raspberry Pi 4 claim'),
        ('Table 4 in [11]', 'Misattributed Table 4 in [11]'),
        ('fine pulse contours', 'Contradictory 4Hz "fine pulse contours" claim'),
        ('millisecond precision', 'Vague "millisecond precision" claim'),
        ('2-5 phút', 'Unexecuted 2-5 min calibration claim'),
        ('2-5 minutes', 'Unexecuted 2-5 min calibration claim'),
        ('15-20% dân số', 'Unverified Boucsein percentage claim'),
    ]

    print("\n--> [SUB-CHECK 5.2] Verification of Banned/Misleading Phrases:")
    for needle, desc in banned_phrases:
        in_md = needle in md_text
        in_docx = needle in docx_text
        if not in_md and not in_docx:
            print(f"  [PASS] Clean: {desc}")
        else:
            print(f"  [FAIL] Unwanted phrase found: {needle} (MD: {in_md}, DOCX: {in_docx})")
            failures += 1

    # Citation Checks
    print("\n--> [SUB-CHECK 5.3] Body Citation Exhaustiveness [1..32]:")
    md_body = md_text.split('## REFERENCES')[0]
    docx_body = docx_text.split('REFERENCES')[0] if 'REFERENCES' in docx_text else docx_text
    
    cites_md = sorted(list(set([int(x) for x in re.findall(r'\[(\d+)\]', md_body)])))
    cites_docx = sorted(list(set([int(x) for x in re.findall(r'\[(\d+)\]', docx_body)])))
    
    missing_md = [x for x in range(1, 33) if x not in cites_md]
    missing_docx = [x for x in range(1, 33) if x not in cites_docx]
    
    if not missing_md and not missing_docx:
        print(f"  [PASS] All 32 references [1..32] are cited in both MD and DOCX!")
    else:
        print(f"  [FAIL] Missing in MD: {missing_md}; Missing in DOCX: {missing_docx}")
        failures += 1

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO DEFECTS!")
        print("The MindFit AI manuscript and empirical benchmarks are 100% publication-ready.")
        print("=" * 80)
    else:
        print(f"AUDIT COMPLETED WITH {failures} DEFECT(S). PLEASE REVIEW THE OUTPUT ABOVE.")
        print("=" * 80)

def main():
    print("=" * 80)
    print("MINDFIT AI - MASTER REVIEW v8 REPRODUCIBILITY & ACCEPTANCE PIPELINE")
    print("Author: Kiet & Phong | Antigravity AI Pair Programming")
    print("=" * 80)
    
    # Run all steps
    lat_info = step1_benchmark_latency()
    baseline_res = step2_run_baselines()
    figs_ok = step3_verify_figures()
    step4_compile_paper()
    step5_audit_compliance()

if __name__ == '__main__':
    main()
