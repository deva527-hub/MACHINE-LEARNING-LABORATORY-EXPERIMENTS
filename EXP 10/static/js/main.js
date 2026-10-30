/**
 * MedVision-ML: Client-Side Dashboard Application
 * Handles pipeline controls, Chart.js visualizations, and asynchronous API calls.
 */

let histogramChart = null;
let featureChart = null;
let rocChart = null;

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSampleSelector();
    initUploadHandler();
    initPipelineActions();
    loadSampleImage("pneumonia", "pneumonia_case_01.jpg"); // Initial default load
});

// -------------------------------------------------------------
// 1. TAB NAVIGATION
// -------------------------------------------------------------
function initTabs() {
    const tabs = document.querySelectorAll(".step-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.dataset.target;
            switchTab(targetId);
        });
    });
}

function switchTab(targetId) {
    document.querySelectorAll(".step-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.target === targetId);
    });
    document.querySelectorAll(".tab-content").forEach(c => {
        c.classList.toggle("hidden", c.id !== targetId);
    });
}

// -------------------------------------------------------------
// 2. SAMPLE SELECTION & IMAGE LOADING
// -------------------------------------------------------------
function initSampleSelector() {
    fetch("/api/samples")
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                renderSampleDropdowns(data.samples);
            }
        })
        .catch(err => console.error("Error loading sample list:", err));
}

function renderSampleDropdowns(samples) {
    const normalContainer = document.getElementById("normal-samples-list");
    const pneumoniaContainer = document.getElementById("pneumonia-samples-list");

    if (normalContainer && samples.normal) {
        normalContainer.innerHTML = samples.normal.map(f => `
            <button onclick="loadSampleImage('normal', '${f}')" class="text-left px-3 py-2 text-xs rounded bg-slate-800/80 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-900/40 transition flex items-center justify-between">
                <span><i class="fa-solid fa-lungs mr-1.5"></i> ${f}</span>
                <span class="text-[10px] bg-emerald-950/80 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-800">Normal</span>
            </button>
        `).join("");
    }

    if (pneumoniaContainer && samples.pneumonia) {
        pneumoniaContainer.innerHTML = samples.pneumonia.map(f => `
            <button onclick="loadSampleImage('pneumonia', '${f}')" class="text-left px-3 py-2 text-xs rounded bg-slate-800/80 hover:bg-rose-600/30 text-rose-300 border border-rose-900/40 transition flex items-center justify-between">
                <span><i class="fa-solid fa-virus mr-1.5"></i> ${f}</span>
                <span class="text-[10px] bg-rose-950/80 text-rose-400 px-1.5 py-0.5 rounded border border-rose-800">Infiltrate</span>
            </button>
        `).join("");
    }
}

function loadSampleImage(category, filename) {
    showNotification(`Loading ${category.toUpperCase()} case: ${filename}...`, "info");
    fetch("/api/load-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, filename })
    })
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                displayLoadedImage(data.metadata, `${category.toUpperCase()}: ${filename}`);
                showNotification(`Loaded ${filename} successfully!`, "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => showNotification("Failed to load sample: " + err, "error"));
}

// -------------------------------------------------------------
// 3. IMAGE UPLOAD HANDLING
// -------------------------------------------------------------
function initUploadHandler() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("image-upload-input");

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-sky-400", "bg-sky-950/20");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-sky-400", "bg-sky-950/20");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-sky-400", "bg-sky-950/20");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

function handleFileUpload(file) {
    showNotification(`Uploading ${file.name}...`, "info");
    const formData = new FormData();
    formData.append("image", file);

    fetch("/api/upload-image", {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                displayLoadedImage(data.metadata, `Uploaded: ${data.filename}`);
                showNotification(`Successfully uploaded ${data.filename}!`, "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => showNotification("Upload failed: " + err, "error"));
}

function displayLoadedImage(metadata, title) {
    const previewImg = document.getElementById("main-image-preview");
    if (previewImg) {
        previewImg.src = metadata.preview_base64;
    }

    const titleEl = document.getElementById("loaded-image-title");
    if (titleEl) titleEl.innerText = title;

    // Update metadata specs
    const metaContainer = document.getElementById("image-metadata-container");
    if (metaContainer) {
        metaContainer.innerHTML = `
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div class="bg-slate-800/80 p-2.5 rounded border border-slate-700">
                    <span class="text-slate-400 block text-[11px]">Dimensions</span>
                    <span class="font-semibold text-sky-300 font-mono">${metadata.width} × ${metadata.height} px</span>
                </div>
                <div class="bg-slate-800/80 p-2.5 rounded border border-slate-700">
                    <span class="text-slate-400 block text-[11px]">Color Mode</span>
                    <span class="font-semibold text-sky-300 font-mono">${metadata.mode} (${metadata.format})</span>
                </div>
                <div class="bg-slate-800/80 p-2.5 rounded border border-slate-700">
                    <span class="text-slate-400 block text-[11px]">Mean Intensity</span>
                    <span class="font-semibold text-emerald-300 font-mono">${metadata.mean_intensity} ± ${metadata.std_intensity}</span>
                </div>
                <div class="bg-slate-800/80 p-2.5 rounded border border-slate-700">
                    <span class="text-slate-400 block text-[11px]">Intensity Range</span>
                    <span class="font-semibold text-amber-300 font-mono">[${metadata.min_intensity}, ${metadata.max_intensity}]</span>
                </div>
            </div>
        `;
    }

    // Render Histogram Chart
    renderHistogram(metadata.histogram_bins, metadata.histogram_counts);
}

function renderHistogram(bins, counts) {
    const ctx = document.getElementById("histogram-chart");
    if (!ctx) return;

    if (histogramChart) {
        histogramChart.destroy();
    }

    histogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins.map(b => `${b}`),
            datasets: [{
                label: 'Pixel Frequency',
                data: counts,
                backgroundColor: 'rgba(56, 189, 248, 0.65)',
                borderColor: 'rgba(56, 189, 248, 1)',
                borderWidth: 1,
                borderRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => `Intensity Level: ${items[0].label}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                }
            }
        }
    });
}

// -------------------------------------------------------------
// 4. PIPELINE ACTIONS & EXECUTION
// -------------------------------------------------------------
function initPipelineActions() {
    // 1-Click Run Entire Pipeline
    const btnRunAll = document.getElementById("btn-run-all");
    if (btnRunAll) {
        btnRunAll.addEventListener("click", runEntirePipeline);
    }

    // Step 2: Run Preprocessor
    const btnPreprocess = document.getElementById("btn-run-preprocess");
    if (btnPreprocess) {
        btnPreprocess.addEventListener("click", runPreprocessing);
    }

    // Step 3: Extract Features
    const btnExtract = document.getElementById("btn-extract-features");
    if (btnExtract) {
        btnExtract.addEventListener("click", runFeatureExtraction);
    }

    // Step 4: Train Model
    const btnTrain = document.getElementById("btn-train-model");
    if (btnTrain) {
        btnTrain.addEventListener("click", runModelTraining);
    }

    // Step 5: Diagnose
    const btnDiagnose = document.getElementById("btn-run-diagnosis");
    if (btnDiagnose) {
        btnDiagnose.addEventListener("click", runDiagnosis);
    }
}

// Execute 1-Click All-in-One Workflow
function runEntirePipeline() {
    setButtonLoading("btn-run-all", true, "Executing Full Pipeline...");
    showNotification("Executing full 5-stage medical ML pipeline...", "info");

    fetch("/api/run-all", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            setButtonLoading("btn-run-all", false, `<i class="fa-solid fa-bolt mr-2"></i> Run Full Pipeline`);
            if (data.status === "success") {
                displayPreprocessingStages(data.preprocessing);
                displayFeatureExtraction(data.features.feature_table, data.features.feature_names, data.features.feature_values);
                displayTrainingEvaluation(data.evaluation);
                displayDiagnosticResult(data.diagnosis);

                switchTab("tab-evaluation");
                showNotification("Pipeline executed successfully! Full diagnosis complete.", "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => {
            setButtonLoading("btn-run-all", false, `<i class="fa-solid fa-bolt mr-2"></i> Run Full Pipeline`);
            showNotification("Pipeline failed: " + err, "error");
        });
}

// Execute Step 2: Preprocessing
function runPreprocessing() {
    setButtonLoading("btn-run-preprocess", true, "Filtering & Enhancing...");
    fetch("/api/preprocess", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            setButtonLoading("btn-run-preprocess", false, `<i class="fa-solid fa-sliders mr-2"></i> Run Preprocessing Pipeline`);
            if (data.status === "success") {
                displayPreprocessingStages(data.stages);
                switchTab("tab-preprocessor");
                showNotification("Image successfully processed through all 5 filter stages!", "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => {
            setButtonLoading("btn-run-preprocess", false, `<i class="fa-solid fa-sliders mr-2"></i> Run Preprocessing Pipeline`);
            showNotification("Preprocessing failed: " + err, "error");
        });
}

function displayPreprocessingStages(stages) {
    const container = document.getElementById("preprocessing-stages-container");
    if (!container) return;

    container.innerHTML = Object.entries(stages).map(([key, stage], idx) => `
        <div class="glass-panel p-4 rounded-xl border border-slate-700/60 hover:border-sky-500/50 transition flex flex-col justify-between">
            <div>
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-semibold px-2 py-0.5 rounded bg-sky-950 text-sky-400 border border-sky-800">Stage ${idx}</span>
                    <span class="text-[11px] text-slate-400 font-mono">128 × 128</span>
                </div>
                <h4 class="text-sm font-semibold text-slate-100 mb-1">${stage.title}</h4>
                <p class="text-xs text-slate-400 mb-3 leading-relaxed">${stage.description}</p>
            </div>
            <div class="medical-img-container rounded-lg aspect-square flex items-center justify-center p-2">
                <img src="${stage.preview}" alt="${stage.title}" class="max-h-full max-w-full object-contain rounded shadow" />
            </div>
        </div>
    `).join("");
}

// Execute Step 3: Feature Extraction
function runFeatureExtraction() {
    setButtonLoading("btn-extract-features", true, "Computing GLCM & Moments...");
    fetch("/api/extract-features", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            setButtonLoading("btn-extract-features", false, `<i class="fa-solid fa-dna mr-2"></i> Extract Clinical Features`);
            if (data.status === "success") {
                displayFeatureExtraction(data.feature_table, data.feature_names, data.feature_values);
                switchTab("tab-features");
                showNotification("Extracted 13 radiological texture, statistical, and edge features!", "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => {
            setButtonLoading("btn-extract-features", false, `<i class="fa-solid fa-dna mr-2"></i> Extract Clinical Features`);
            showNotification("Feature extraction failed: " + err, "error");
        });
}

function displayFeatureExtraction(table, names, values) {
    const tbody = document.getElementById("features-table-body");
    if (tbody) {
        tbody.innerHTML = table.map(f => `
            <tr class="border-b border-slate-800 hover:bg-slate-800/40 transition">
                <td class="py-2.5 px-3 font-mono text-sky-300 font-semibold">${f.name}</td>
                <td class="py-2.5 px-3 font-mono text-emerald-400 font-bold">${f.value}</td>
                <td class="py-2.5 px-3 text-slate-400 text-xs">${f.description}</td>
            </tr>
        `).join("");
    }

    renderFeatureChart(names, values);
}

function renderFeatureChart(names, values) {
    const ctx = document.getElementById("features-bar-chart");
    if (!ctx) return;

    if (featureChart) featureChart.destroy();

    // Normalize values between 0 and 1 for comparative radar/bar display
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const normalized = values.map(v => (v - minVal) / (maxVal - minVal + 1e-8));

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names,
            datasets: [{
                label: 'Relative Feature Magnitude',
                data: normalized,
                backgroundColor: 'rgba(14, 165, 233, 0.7)',
                borderColor: '#0284c7',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Raw Value: ${values[ctx.dataIndex]}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { family: 'JetBrains Mono', size: 11 } }
                }
            }
        }
    });
}

// Execute Step 4: Model Training
function runModelTraining() {
    const algo = document.getElementById("select-algorithm")?.value || "svm";
    const testSplit = parseFloat(document.getElementById("slider-test-split")?.value || "0.25");

    setButtonLoading("btn-train-model", true, `Training ${algo.toUpperCase()}...`);
    showNotification(`Training ${algo.toUpperCase()} model on clinical dataset...`, "info");

    fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ algorithm: algo, test_size: testSplit })
    })
        .then(res => res.json())
        .then(data => {
            setButtonLoading("btn-train-model", false, `<i class="fa-solid fa-graduation-cap mr-2"></i> Train Classifier`);
            if (data.status === "success") {
                displayTrainingEvaluation(data.report);
                switchTab("tab-training");
                showNotification(`Model trained successfully! Accuracy: ${data.report.metrics.accuracy}%`, "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => {
            setButtonLoading("btn-train-model", false, `<i class="fa-solid fa-graduation-cap mr-2"></i> Train Classifier`);
            showNotification("Training failed: " + err, "error");
        });
}

function displayTrainingEvaluation(report) {
    // 1. Metric Badges
    const m = report.metrics;
    document.getElementById("metric-accuracy").innerText = `${m.accuracy}%`;
    document.getElementById("metric-recall").innerText = `${m.sensitivity_recall}%`;
    document.getElementById("metric-specificity").innerText = `${m.specificity}%`;
    document.getElementById("metric-precision").innerText = `${m.precision}%`;
    document.getElementById("metric-f1").innerText = `${m.f1_score}%`;
    document.getElementById("metric-auc").innerText = `${m.roc_auc}`;

    // 2. Training details
    const modelDetails = document.getElementById("training-details-summary");
    if (modelDetails) {
        modelDetails.innerHTML = `
            <div class="flex flex-wrap gap-4 text-xs font-mono text-slate-300">
                <span><b class="text-slate-400">Model:</b> <span class="text-sky-300">${report.algorithm}</span></span>
                <span><b class="text-slate-400">Total Samples:</b> ${report.total_samples}</span>
                <span><b class="text-slate-400">Train Set:</b> ${report.train_samples}</span>
                <span><b class="text-slate-400">Test Set:</b> ${report.test_samples}</span>
            </div>
        `;
    }

    // 3. Confusion Matrix
    const cm = report.confusion_matrix;
    document.getElementById("cm-tp").innerText = cm.tp;
    document.getElementById("cm-tn").innerText = cm.tn;
    document.getElementById("cm-fp").innerText = cm.fp;
    document.getElementById("cm-fn").innerText = cm.fn;

    // 4. ROC Curve Chart
    if (report.roc_curve) {
        renderRocCurve(report.roc_curve.fpr, report.roc_curve.tpr, m.roc_auc);
    }
}

function renderRocCurve(fpr, tpr, aucScore) {
    const ctx = document.getElementById("roc-curve-chart");
    if (!ctx) return;

    if (rocChart) rocChart.destroy();

    const dataPoints = fpr.map((x, idx) => ({ x: x, y: tpr[idx] }));

    rocChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `ROC Curve (AUC = ${aucScore})`,
                    data: dataPoints,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'Random Guess (AUC = 0.50)',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: 'rgba(255, 255, 255, 0.3)',
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#cbd5e1', font: { size: 11 } } }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'True Positive Rate (Sensitivity)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Execute Step 5: Live Diagnosis
function runDiagnosis() {
    setButtonLoading("btn-run-diagnosis", true, "Evaluating Clinical Features...");
    fetch("/api/diagnose", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            setButtonLoading("btn-run-diagnosis", false, `<i class="fa-solid fa-stethoscope mr-2"></i> Run Clinical Diagnosis`);
            if (data.status === "success") {
                displayDiagnosticResult(data.result);
                switchTab("tab-evaluation");
                showNotification(`Diagnosis Complete: ${data.result.prediction}`, "success");
            } else {
                showNotification(data.message, "error");
            }
        })
        .catch(err => {
            setButtonLoading("btn-run-diagnosis", false, `<i class="fa-solid fa-stethoscope mr-2"></i> Run Clinical Diagnosis`);
            showNotification("Diagnosis failed: " + err, "error");
        });
}

function displayDiagnosticResult(res) {
    const verdictEl = document.getElementById("diag-verdict");
    const confidenceEl = document.getElementById("diag-confidence");
    const riskBadge = document.getElementById("diag-risk-badge");
    const probNormalEl = document.getElementById("diag-prob-normal");
    const probPneumoniaEl = document.getElementById("diag-prob-pneumonia");
    const barNormal = document.getElementById("diag-bar-normal");
    const barPneumonia = document.getElementById("diag-bar-pneumonia");
    const findingsEl = document.getElementById("diag-findings");
    const recommendationEl = document.getElementById("diag-recommendation");
    const diagCard = document.getElementById("diagnosis-summary-card");

    const isPneumonia = res.prediction === "PNEUMONIA";

    if (verdictEl) {
        verdictEl.innerText = isPneumonia ? "PNEUMONIA DETECTED" : "NORMAL (CLEAR LUNG)";
        verdictEl.className = `text-2xl font-bold font-mono ${isPneumonia ? "text-rose-400" : "text-emerald-400"}`;
    }

    if (confidenceEl) confidenceEl.innerText = `${res.confidence}%`;

    if (riskBadge) {
        riskBadge.innerText = res.risk_level;
        riskBadge.className = `px-3 py-1 rounded-full text-xs font-semibold ${isPneumonia ? "bg-rose-950 text-rose-300 border border-rose-700" : "bg-emerald-950 text-emerald-300 border border-emerald-700"}`;
    }

    if (probNormalEl) probNormalEl.innerText = `${res.probabilities.NORMAL}%`;
    if (probPneumoniaEl) probPneumoniaEl.innerText = `${res.probabilities.PNEUMONIA}%`;

    if (barNormal) barNormal.style.width = `${res.probabilities.NORMAL}%`;
    if (barPneumonia) barPneumonia.style.width = `${res.probabilities.PNEUMONIA}%`;

    if (findingsEl) findingsEl.innerText = res.radiological_finding;
    if (recommendationEl) recommendationEl.innerText = res.recommendation;

    if (diagCard) {
        diagCard.classList.remove("glow-success", "glow-danger");
        diagCard.classList.add(isPneumonia ? "glow-danger" : "glow-success");
    }
}

// -------------------------------------------------------------
// UI UTILITIES & NOTIFICATIONS
// -------------------------------------------------------------
function setButtonLoading(buttonId, isLoading, text) {
    const btn = document.getElementById(buttonId);
    if (!btn) return;
    btn.disabled = isLoading;
    if (isLoading) {
        btn.innerHTML = `<span class="spinner inline-block mr-2"></span> ${text}`;
        btn.classList.add("opacity-75", "cursor-not-allowed");
    } else {
        btn.innerHTML = text;
        btn.classList.remove("opacity-75", "cursor-not-allowed");
    }
}

function showNotification(message, type = "info") {
    const toast = document.getElementById("notification-toast");
    if (!toast) return;

    const iconMap = {
        success: '<i class="fa-solid fa-circle-check text-emerald-400 mr-2 text-lg"></i>',
        error: '<i class="fa-solid fa-triangle-exclamation text-rose-400 mr-2 text-lg"></i>',
        info: '<i class="fa-solid fa-circle-info text-sky-400 mr-2 text-lg"></i>'
    };

    toast.innerHTML = `
        <div class="flex items-center">
            ${iconMap[type] || iconMap.info}
            <span class="text-xs text-slate-100 font-medium">${message}</span>
        </div>
    `;

    toast.classList.remove("translate-y-24", "opacity-0");
    toast.classList.add("translate-y-0", "opacity-100");

    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => {
        toast.classList.remove("translate-y-0", "opacity-100");
        toast.classList.add("translate-y-24", "opacity-0");
    }, 3800);
}

