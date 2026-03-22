/* ═══════════════════════════════════════════════════════════
   FraudSentinel — Dashboard Logic
   Handles: API calls, model cards, ROC/PR charts,
            confusion matrix, custom predict form
   ═══════════════════════════════════════════════════════════ */

const API_BASE = 'http://localhost:5000';

// ─── PALETTE ─────────────────────────────────────────────────────────────────
const MODEL_COLORS = {
  'Logistic Regression': { line: '#00d4ff', fill: 'rgba(0,212,255,0.1)' },
  'SVM':                 { line: '#ff4b5c', fill: 'rgba(255,75,92,0.1)' },
  'Decision Tree':       { line: '#ffc93c', fill: 'rgba(255,201,60,0.1)' },
  'Neural Network':      { line: '#00e896', fill: 'rgba(0,232,150,0.1)' }
};

const CHART_DEFAULTS = {
  color: '#7a8098',
  gridColor: 'rgba(255,255,255,0.04)',
  font: 'DM Mono'
};

// ─── CHART.JS GLOBAL DEFAULTS ────────────────────────────────────────────────
Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font.family = CHART_DEFAULTS.font;
Chart.defaults.font.size = 11;

// ─── STATE ───────────────────────────────────────────────────────────────────
let metricsData  = null;
let rocChart     = null;
let prChart      = null;

// ─── UTILS ───────────────────────────────────────────────────────────────────
function fmt(v) { return (v * 100).toFixed(2) + '%'; }
function fmtNum(v) { return (v * 100).toFixed(1); }

async function apiFetch(path) {
  const res  = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${path}`);
  return res.json();
}

// Count-up animation
function countUp(element, target, duration = 1200) {
  const start     = performance.now();
  const targetVal = parseFloat(target);
  const isPercent = typeof target === 'string' && target.includes('%');

  function tick(now) {
    const elapsed  = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const ease     = 1 - Math.pow(1 - progress, 3);
    const current  = targetVal * ease;

    if (isPercent) {
      element.textContent = current.toFixed(1) + '%';
    } else {
      element.textContent = current.toFixed(4);
    }

    if (progress < 1) requestAnimationFrame(tick);
    else element.textContent = target;
  }
  requestAnimationFrame(tick);
}

// ─── API STATUS ──────────────────────────────────────────────────────────────
async function checkApiStatus() {
  const dot   = document.getElementById('apiStatusDot');
  const label = document.getElementById('apiStatusLabel');

  try {
    const data = await apiFetch('/api/health');
    dot.className   = 'status-dot ' + (data.demo_mode ? 'demo' : 'online');
    label.textContent = data.demo_mode ? 'Demo Mode' : 'API Online';
  } catch {
    dot.className   = 'status-dot offline';
    label.textContent = 'Demo Mode';
  }
}

// ─── MODEL CARDS ─────────────────────────────────────────────────────────────
function renderModelCards(models, bestModelName) {
  const container = document.getElementById('modelCards');
  container.innerHTML = '';

  models.forEach((m, i) => {
    const isBest  = m.name === bestModelName;
    const color   = MODEL_COLORS[m.name] || { line: '#00d4ff' };
    const metrics = [
      { key: 'Accuracy',  val: m.accuracy,  cls: 'acc' },
      { key: 'Precision', val: m.precision, cls: 'prec' },
      { key: 'Recall',    val: m.recall,    cls: 'recall' },
      { key: 'F1 Score',  val: m.f1_score,  cls: 'f1' }
    ];

    const card = document.createElement('div');
    card.className = 'model-card' + (isBest ? ' best' : '');
    card.style.animationDelay = (i * 0.1) + 's';
    card.style.borderTopColor = color.line;

    card.innerHTML = `
      <div class="card-model-name">${m.name}</div>
      <div class="card-auc" id="auc_${i}">0.0000</div>
      <div class="card-auc-label">ROC-AUC Score</div>
      <div class="card-metrics">
        ${metrics.map(metric => `
          <div class="card-metric-row">
            <span class="metric-name">${metric.key}</span>
            <div class="metric-bar-wrap">
              <div class="metric-bar-track">
                <div class="metric-bar-fill ${metric.cls}"
                     id="bar_${i}_${metric.cls}"
                     data-target="${metric.val}">
                </div>
              </div>
            </div>
            <span class="metric-val" id="mval_${i}_${metric.cls}">0.00%</span>
          </div>
        `).join('')}
      </div>
    `;

    container.appendChild(card);

    // Animate after paint
    setTimeout(() => {
      card.classList.add('fade-in');

      // AUC count-up
      countUp(document.getElementById(`auc_${i}`), m.roc_auc.toFixed(4), 1400);

      // Metric bars
      metrics.forEach(metric => {
        const bar = document.getElementById(`bar_${i}_${metric.cls}`);
        const val = document.getElementById(`mval_${i}_${metric.cls}`);

        if (bar) {
          setTimeout(() => {
            bar.style.width = (metric.val * 100) + '%';
          }, 200);
        }
        if (val) {
          setTimeout(() => {
            countUp(val, (metric.val * 100).toFixed(1) + '%', 1200);
          }, 300);
        }
      });
    }, i * 120);
  });
}

// ─── ROC CHART ───────────────────────────────────────────────────────────────
function renderROCChart(models) {
  const ctx = document.getElementById('rocChart').getContext('2d');

  const datasets = models.map(m => {
    const color = MODEL_COLORS[m.name] || { line: '#888' };
    return {
      label:       m.name + ` (AUC=${m.roc_auc.toFixed(3)})`,
      data:        m.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: m.roc_curve.tpr[i] })),
      borderColor: color.line,
      backgroundColor: color.fill,
      borderWidth: 2,
      pointRadius: 0,
      tension:     0.3,
      fill:        false
    };
  });

  // Random classifier diagonal
  datasets.push({
    label: 'Random Classifier',
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: 'rgba(255,255,255,0.2)',
    borderWidth: 1,
    borderDash: [5, 5],
    pointRadius: 0,
    fill: false
  });

  if (rocChart) rocChart.destroy();

  rocChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000 },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            boxWidth: 12,
            padding: 12,
            font: { size: 11, family: 'DM Mono' }
          }
        },
        tooltip: {
          backgroundColor: '#0d1224',
          borderColor: 'rgba(0,212,255,0.2)',
          borderWidth: 1,
          titleFont: { family: 'DM Mono', size: 11 },
          bodyFont:  { family: 'DM Mono', size: 11 },
          callbacks: {
            label: ctx => ` FPR=${ctx.parsed.x.toFixed(3)} | TPR=${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'False Positive Rate', font: { size: 11 }, color: '#7a8098' },
          min: 0, max: 1,
          grid: { color: CHART_DEFAULTS.gridColor },
          ticks: { maxTicksLimit: 6 }
        },
        y: {
          title: { display: true, text: 'True Positive Rate', font: { size: 11 }, color: '#7a8098' },
          min: 0, max: 1,
          grid: { color: CHART_DEFAULTS.gridColor },
          ticks: { maxTicksLimit: 6 }
        }
      }
    }
  });
}

// ─── PR CHART ────────────────────────────────────────────────────────────────
function renderPRChart(models) {
  const ctx = document.getElementById('prChart').getContext('2d');

  const datasets = models.map(m => {
    const color = MODEL_COLORS[m.name] || { line: '#888' };
    return {
      label:       m.name + ` (AP=${m.avg_precision.toFixed(3)})`,
      data:        m.pr_curve.recall.map((r, i) => ({ x: r, y: m.pr_curve.precision[i] })),
      borderColor: color.line,
      backgroundColor: color.fill,
      borderWidth: 2,
      pointRadius: 0,
      tension:     0.3,
      fill:        false
    };
  });

  if (prChart) prChart.destroy();

  prChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000 },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            boxWidth: 12,
            padding: 12,
            font: { size: 11, family: 'DM Mono' }
          }
        },
        tooltip: {
          backgroundColor: '#0d1224',
          borderColor: 'rgba(255,75,92,0.2)',
          borderWidth: 1,
          titleFont: { family: 'DM Mono', size: 11 },
          bodyFont:  { family: 'DM Mono', size: 11 },
          callbacks: {
            label: ctx => ` Recall=${ctx.parsed.x.toFixed(3)} | Precision=${ctx.parsed.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Recall', font: { size: 11 }, color: '#7a8098' },
          min: 0, max: 1,
          grid: { color: CHART_DEFAULTS.gridColor },
          ticks: { maxTicksLimit: 6 }
        },
        y: {
          title: { display: true, text: 'Precision', font: { size: 11 }, color: '#7a8098' },
          min: 0, max: 1,
          grid: { color: CHART_DEFAULTS.gridColor },
          ticks: { maxTicksLimit: 6 }
        }
      }
    }
  });
}

// ─── CONFUSION MATRIX ────────────────────────────────────────────────────────
function renderConfusionMatrix(modelName) {
  const wrap = document.getElementById('confusionMatrix');
  wrap.innerHTML = '<div class="matrix-loading">Loading...</div>';

  apiFetch(`/api/confusion_matrix/${modelName}`)
    .then(data => {
      const [[tn, fp], [fn, tp]] = data.matrix;
      const total    = tn + fp + fn + tp;
      const accuracy = ((tn + tp) / total * 100).toFixed(2);

      wrap.innerHTML = `
        <div class="matrix-axis-label" style="margin-bottom:8px">Predicted →</div>
        <div class="matrix-grid fade-in">
          <div class="matrix-cell tn">
            <div class="matrix-cell-type">True Negative</div>
            <div class="matrix-cell-value" id="mcTN">0</div>
            <div class="matrix-cell-desc">Correctly rejected</div>
          </div>
          <div class="matrix-cell fp">
            <div class="matrix-cell-type">False Positive</div>
            <div class="matrix-cell-value" id="mcFP">0</div>
            <div class="matrix-cell-desc">False alarm</div>
          </div>
          <div class="matrix-cell fn">
            <div class="matrix-cell-type">False Negative</div>
            <div class="matrix-cell-value" id="mcFN">0</div>
            <div class="matrix-cell-desc">Missed fraud</div>
          </div>
          <div class="matrix-cell tp">
            <div class="matrix-cell-type">True Positive</div>
            <div class="matrix-cell-value" id="mcTP">0</div>
            <div class="matrix-cell-desc">Caught fraud</div>
          </div>
        </div>
      `;

      // Animate counts
      setTimeout(() => {
        countUp(document.getElementById('mcTN'), tn.toLocaleString(), 800);
        countUp(document.getElementById('mcFP'), fp.toLocaleString(), 800);
        countUp(document.getElementById('mcFN'), fn.toLocaleString(), 800);
        countUp(document.getElementById('mcTP'), tp.toLocaleString(), 800);
      }, 100);

      // Stats panel
      const statsDiv  = document.getElementById('confusionStats');
      const precision = tp / (tp + fp) || 0;
      const recall    = tp / (tp + fn) || 0;
      const f1        = (2 * precision * recall) / (precision + recall) || 0;

      statsDiv.innerHTML = `
        <div class="cstat-item">
          <div class="cstat-label">Accuracy</div>
          <div class="cstat-value" style="color:var(--blue)">${accuracy}%</div>
        </div>
        <div class="cstat-item">
          <div class="cstat-label">Precision</div>
          <div class="cstat-value" style="color:var(--green)">${(precision*100).toFixed(2)}%</div>
        </div>
        <div class="cstat-item">
          <div class="cstat-label">Recall</div>
          <div class="cstat-value" style="color:var(--yellow)">${(recall*100).toFixed(2)}%</div>
        </div>
        <div class="cstat-item">
          <div class="cstat-label">F1 Score</div>
          <div class="cstat-value" style="color:var(--red)">${(f1*100).toFixed(2)}%</div>
        </div>
        <div class="cstat-item">
          <div class="cstat-label">Total Tested</div>
          <div class="cstat-value" style="color:var(--text-secondary)">${total.toLocaleString()}</div>
        </div>
      `;
    })
    .catch(err => {
      wrap.innerHTML = `<div class="matrix-loading">Using demo data...</div>`;
      // Fallback to demo data
      const demo = {
        logistic_regression: [[56851, 11], [30, 61]],
        svm:                 [[56857, 5],  [22, 69]],
        decision_tree:       [[56836, 26], [23, 68]],
        neural_network:      [[56858, 4],  [17, 74]]
      };
      const key    = modelName.toLowerCase().replace(/ /g, '_');
      const matrix = demo[key] || demo.neural_network;
      renderConfusionMatrix._lastData = { matrix };
      setTimeout(() => {
        const fakeData = { matrix, model: modelName };
        const [[tn, fp], [fn, tp]] = matrix;
        const total    = tn + fp + fn + tp;
        const accuracy = ((tn + tp) / total * 100).toFixed(2);

        wrap.innerHTML = `
          <div class="matrix-axis-label" style="margin-bottom:8px">Predicted →</div>
          <div class="matrix-grid fade-in">
            <div class="matrix-cell tn">
              <div class="matrix-cell-type">True Negative</div>
              <div class="matrix-cell-value">${tn.toLocaleString()}</div>
              <div class="matrix-cell-desc">Correctly rejected</div>
            </div>
            <div class="matrix-cell fp">
              <div class="matrix-cell-type">False Positive</div>
              <div class="matrix-cell-value">${fp.toLocaleString()}</div>
              <div class="matrix-cell-desc">False alarm</div>
            </div>
            <div class="matrix-cell fn">
              <div class="matrix-cell-type">False Negative</div>
              <div class="matrix-cell-value">${fn.toLocaleString()}</div>
              <div class="matrix-cell-desc">Missed fraud</div>
            </div>
            <div class="matrix-cell tp">
              <div class="matrix-cell-type">True Positive</div>
              <div class="matrix-cell-value">${tp.toLocaleString()}</div>
              <div class="matrix-cell-desc">Caught fraud</div>
            </div>
          </div>
        `;

        const statsDiv  = document.getElementById('confusionStats');
        const precision = tp / (tp + fp) || 0;
        const recall    = tp / (tp + fn) || 0;
        const f1        = (2 * precision * recall) / (precision + recall) || 0;
        statsDiv.innerHTML = `
          <div class="cstat-item"><div class="cstat-label">Accuracy</div><div class="cstat-value">${accuracy}%</div></div>
          <div class="cstat-item"><div class="cstat-label">Precision</div><div class="cstat-value" style="color:var(--green)">${(precision*100).toFixed(2)}%</div></div>
          <div class="cstat-item"><div class="cstat-label">Recall</div><div class="cstat-value" style="color:var(--yellow)">${(recall*100).toFixed(2)}%</div></div>
          <div class="cstat-item"><div class="cstat-label">F1 Score</div><div class="cstat-value" style="color:var(--red)">${(f1*100).toFixed(2)}%</div></div>
          <div class="cstat-item"><div class="cstat-label">Total Tested</div><div class="cstat-value" style="color:var(--text-secondary)">${total.toLocaleString()}</div></div>
        `;
      }, 200);
    });
}

// ─── PREDICT FORM ────────────────────────────────────────────────────────────
const SLIDER_MAP = [
  { id: 'inp_amount', displayId: 'amountVal', key: 'Amount', fmt: v => '$' + parseFloat(v).toFixed(2) },
  { id: 'inp_time',   displayId: 'timeVal',   key: 'Time',   fmt: v => parseInt(v).toLocaleString() + 's' },
  { id: 'inp_v1',     displayId: 'v1Val',     key: 'V1',     fmt: v => parseFloat(v).toFixed(2) },
  { id: 'inp_v4',     displayId: 'v4Val',     key: 'V4',     fmt: v => parseFloat(v).toFixed(2) },
  { id: 'inp_v10',    displayId: 'v10Val',    key: 'V10',    fmt: v => parseFloat(v).toFixed(2) },
  { id: 'inp_v14',    displayId: 'v14Val',    key: 'V14',    fmt: v => parseFloat(v).toFixed(2) },
  { id: 'inp_v17',    displayId: 'v17Val',    key: 'V17',    fmt: v => parseFloat(v).toFixed(2) },
];

// Wire sliders
SLIDER_MAP.forEach(({ id, displayId, fmt: fmtFn }) => {
  const inp  = document.getElementById(id);
  const disp = document.getElementById(displayId);
  if (inp && disp) {
    inp.addEventListener('input', () => {
      disp.textContent = fmtFn(inp.value);
    });
  }
});

// Preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const preset = btn.dataset.preset;
    if (preset === 'safe') {
      document.getElementById('inp_amount').value = 45;
      document.getElementById('inp_time').value   = 86400;
      document.getElementById('inp_v1').value     = 1.2;
      document.getElementById('inp_v4').value     = 0.8;
      document.getElementById('inp_v10').value    = 0.5;
      document.getElementById('inp_v14').value    = -0.3;
      document.getElementById('inp_v17').value    = -0.8;
    } else {
      document.getElementById('inp_amount').value = 2350;
      document.getElementById('inp_time').value   = 11000;
      document.getElementById('inp_v1').value     = -3.5;
      document.getElementById('inp_v4').value     = 4.2;
      document.getElementById('inp_v10').value    = -4.8;
      document.getElementById('inp_v14').value    = -6.1;
      document.getElementById('inp_v17').value    = -8.5;
    }
    // Update displays
    SLIDER_MAP.forEach(({ id, displayId, fmt: fmtFn }) => {
      const inp  = document.getElementById(id);
      const disp = document.getElementById(displayId);
      if (inp && disp) disp.textContent = fmtFn(inp.value);
    });
  });
});

function getTransactionPayload() {
  const payload = {};
  // Set all V features to 0 by default
  for (let i = 1; i <= 28; i++) payload[`V${i}`] = 0;

  SLIDER_MAP.forEach(({ id, key }) => {
    const inp = document.getElementById(id);
    if (inp) payload[key] = parseFloat(inp.value);
  });

  return payload;
}

function getColorClass(prob) {
  if (prob < 0.3) return 'prob-safe';
  if (prob < 0.6) return 'prob-warning';
  return 'prob-danger';
}

function getColorHex(prob) {
  if (prob < 0.3) return '#00e896';
  if (prob < 0.6) return '#ffc93c';
  return '#ff4b5c';
}

function renderPredictResult(data) {
  const panel   = document.getElementById('predictResult');
  const isFraud = data.verdict === 'FRAUD';
  const maxProb = (data.max_fraud_probability * 100).toFixed(1);

  panel.innerHTML = `
    <div class="result-verdict">
      <div class="result-verdict-badge ${isFraud ? 'fraud' : 'legit'}">
        ${isFraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE'}
      </div>
      <div class="result-max-prob">
        Maximum fraud probability: <strong>${maxProb}%</strong>
      </div>
    </div>
    <div class="result-models">
      ${Object.entries(data.predictions).map(([name, pred]) => {
        const prob     = pred.probability;
        const pct      = (prob * 100).toFixed(1);
        const colorCls = getColorClass(prob);
        const colorHex = getColorHex(prob);
        return `
          <div class="result-model-row fade-in">
            <div class="result-model-header">
              <span class="result-model-name">${name}</span>
              <span class="result-model-prob" style="color:${colorHex}">${pct}%</span>
            </div>
            <div class="result-bar-track">
              <div class="result-bar-fill ${colorCls}"
                   id="rb_${name.replace(/\s/g,'_')}"
                   data-target="${prob * 100}">
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;

  // Animate bars
  setTimeout(() => {
    Object.keys(data.predictions).forEach(name => {
      const bar = document.getElementById('rb_' + name.replace(/\s/g, '_'));
      if (bar) bar.style.width = bar.dataset.target + '%';
    });
  }, 100);
}

// Predict button
document.getElementById('predictBtn').addEventListener('click', async () => {
  const btn   = document.getElementById('predictBtn');
  const panel = document.getElementById('predictResult');

  btn.disabled   = true;
  btn.innerHTML  = '<span class="loading-spinner"></span> Analyzing...';
  panel.innerHTML = '<div class="result-idle"><div class="loading-spinner"></div><div class="result-idle-text" style="margin-top:16px">Running all models...</div></div>';

  const payload = getTransactionPayload();

  try {
    const data = await fetch(API_BASE + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    renderPredictResult(data);
  } catch (err) {
    // Fallback: simulate a result
    const isLikelyFraud = Math.abs(payload.V14) > 4 || Math.abs(payload.V10) > 3 || payload.Amount > 2000;
    const baseFraudProb = isLikelyFraud ? 0.7 + Math.random() * 0.25 : 0.02 + Math.random() * 0.1;

    const fakeData = {
      verdict: baseFraudProb > 0.5 ? 'FRAUD' : 'LEGITIMATE',
      max_fraud_probability: baseFraudProb,
      predictions: {
        'Logistic Regression': { probability: Math.min(1, baseFraudProb + (Math.random() - 0.5) * 0.2), prediction: baseFraudProb > 0.5 ? 1 : 0 },
        'SVM':                 { probability: Math.min(1, baseFraudProb + (Math.random() - 0.5) * 0.15), prediction: baseFraudProb > 0.5 ? 1 : 0 },
        'Decision Tree':       { probability: Math.min(1, baseFraudProb + (Math.random() - 0.5) * 0.25), prediction: baseFraudProb > 0.5 ? 1 : 0 },
        'Neural Network':      { probability: Math.min(1, baseFraudProb + (Math.random() - 0.5) * 0.1), prediction: baseFraudProb > 0.5 ? 1 : 0 }
      }
    };

    renderPredictResult(fakeData);
  }

  btn.disabled  = false;
  btn.innerHTML = '<span class="predict-btn-icon">⚡</span> Analyze Transaction';
});

// ─── CONFUSION MATRIX DROPDOWN ────────────────────────────────────────────────
document.getElementById('modelSelector').addEventListener('change', e => {
  renderConfusionMatrix(e.target.value);
});

// ─── MAIN INIT ───────────────────────────────────────────────────────────────
async function init() {
  await checkApiStatus();

  let metrics = null;
  try {
    metrics = await apiFetch('/api/metrics');
  } catch {
    // Use embedded demo metrics
    metrics = getDemoMetrics();
  }

  metricsData = metrics;

  renderModelCards(metrics.models, metrics.best_model);
  renderROCChart(metrics.models);
  renderPRChart(metrics.models);
  renderConfusionMatrix('neural_network');
}

// Demo metrics fallback
function getDemoMetrics() {
  return {
    best_model: 'Neural Network',
    models: [
      {
        name: 'Logistic Regression',
        accuracy: 0.9991, precision: 0.8621, recall: 0.6735, f1_score: 0.7561, roc_auc: 0.9712,
        avg_precision: 0.7234,
        confusion_matrix: [[56851, 11], [30, 61]],
        roc_curve: { fpr: [0,0.002,0.005,0.01,0.05,0.1,0.2,0.5,1], tpr: [0,0.55,0.72,0.82,0.89,0.92,0.94,0.97,1] },
        pr_curve:  { precision: [1,0.95,0.90,0.85,0.75,0.65,0.50,0.10], recall: [0,0.30,0.50,0.62,0.72,0.80,0.88,1] }
      },
      {
        name: 'SVM',
        accuracy: 0.9993, precision: 0.9024, recall: 0.7551, f1_score: 0.8222, roc_auc: 0.9841,
        avg_precision: 0.8102,
        confusion_matrix: [[56857, 5], [22, 69]],
        roc_curve: { fpr: [0,0.001,0.003,0.008,0.03,0.08,0.15,0.4,1], tpr: [0,0.62,0.78,0.85,0.91,0.94,0.96,0.98,1] },
        pr_curve:  { precision: [1,0.97,0.93,0.90,0.82,0.72,0.56,0.10], recall: [0,0.35,0.55,0.68,0.77,0.84,0.91,1] }
      },
      {
        name: 'Decision Tree',
        accuracy: 0.9987, precision: 0.7302, recall: 0.7551, f1_score: 0.7424, roc_auc: 0.9201,
        avg_precision: 0.6891,
        confusion_matrix: [[56836, 26], [23, 68]],
        roc_curve: { fpr: [0,0.003,0.007,0.015,0.06,0.12,0.25,0.55,1], tpr: [0,0.48,0.65,0.75,0.83,0.88,0.91,0.95,1] },
        pr_curve:  { precision: [1,0.90,0.83,0.77,0.68,0.58,0.45,0.10], recall: [0,0.28,0.46,0.60,0.70,0.78,0.86,1] }
      },
      {
        name: 'Neural Network',
        accuracy: 0.9994, precision: 0.9143, recall: 0.8163, f1_score: 0.8625, roc_auc: 0.9912,
        avg_precision: 0.8834,
        confusion_matrix: [[56858, 4], [17, 74]],
        roc_curve: { fpr: [0,0.0005,0.002,0.006,0.02,0.06,0.12,0.35,1], tpr: [0,0.70,0.82,0.88,0.93,0.96,0.97,0.99,1] },
        pr_curve:  { precision: [1,0.98,0.95,0.92,0.86,0.77,0.61,0.10], recall: [0,0.40,0.60,0.72,0.81,0.87,0.93,1] }
      }
    ]
  };
}

// Boot
window.addEventListener('DOMContentLoaded', init);
