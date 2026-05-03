const fmt = new Intl.NumberFormat("en-IN");
const pct = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 });
const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

const state = {
  metrics: null,
  forecasts: [],
  zones: [],
  anomalies: [],
  evidence: {},
  pipeline: null,
  theftValidation: null,
  selectedFeeder: null,
  selectedMeter: null,
  charts: {},
};

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} failed with ${res.status}`);
  return res.json();
}

function byId(id) { return document.getElementById(id); }

function confidenceClass(value) { return `${String(value || "Low").replace(" ", "")}Confidence`; }

/* ─── Dark Mode ─── */
function initTheme() {
  const saved = localStorage.getItem("gs-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "");
  updateThemeButton();
}
function toggleTheme() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  document.documentElement.setAttribute("data-theme", isLight ? "" : "light");
  localStorage.setItem("gs-theme", isLight ? "dark" : "light");
  updateThemeButton();
  if (state.forecasts.length) renderForecast();
  renderTheftValidation();
}
function updateThemeButton() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  byId("darkToggle").textContent = isLight ? "◑ Dark Mode" : "◐ Light Mode";
}
function isDark() { return document.documentElement.getAttribute("data-theme") !== "light"; }
function chartColors() {
  return {
    text: isDark() ? "#8892a4" : "#5a6478",
    grid: isDark() ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
    accent: "#22d3a7",
    warn: "#f59e0b",
    risk: "#ef4444",
    ok: "#22c55e",
    accentFill: isDark() ? "rgba(34,211,167,0.15)" : "rgba(13,155,122,0.1)",
  };
}

/* ─── Metrics Cards ─── */
function renderMetrics() {
  const m = state.metrics;
  const smape = (m.forecast_smape || m.forecast_mape);
  const cards = [
    ["Dataset", "Real", m.dataset_source],
    ["Rows", fmt.format(m.meter_rows), `${m.data_granularity_minutes}-min readings`],
    ["Forecast sMAPE", `${pct.format(smape * 100)}%`, `MAE ${m.forecast_mae_kw} kW`],
    ["Meters / Feeders", `${m.meters}/${m.feeders}`, "Locality groups"],
    ["Anomaly Mode", m.theft_f1 ? `F1 ${m.theft_f1.toFixed(2)}` : "Ops", m.anomaly_mode?.split(" ").slice(0, 4).join(" ")],
    ["SGCC Theft F1", m.sgcc_theft_f1 ? m.sgcc_theft_f1.toFixed(3) : "N/A", m.sgcc_theft_validation ? "Labelled validation ✓" : "Pending"],
  ];
  byId("metrics").innerHTML = cards
    .map(([label, value, hint]) => `<article class="card"><span>${label}</span><b>${value}</b><span class="hint">${hint}</span></article>`)
    .join("");
  byId("lastUpdated").textContent = `${new Date(m.generated_at).toLocaleString()}`;
}

/* ─── Forecast Chart (Chart.js) ─── */
function uniqueFeeders() { return [...new Set(state.forecasts.map(d => d.feeder_id))].sort(); }

function setupControls() {
  const feeders = uniqueFeeders();
  if (!state.selectedFeeder || !feeders.includes(state.selectedFeeder)) state.selectedFeeder = feeders[0];
  byId("feederSelect").innerHTML = feeders.map(id => `<option value="${id}">${id}</option>`).join("");
  byId("feederSelect").value = state.selectedFeeder;
}

function renderForecast() {
  const feederRows = state.forecasts.filter(d => d.feeder_id === state.selectedFeeder);
  if (!feederRows.length) return;
  const firstTs = new Date(feederRows[0].timestamp);
  const secondTs = new Date(feederRows[1]?.timestamp || firstTs);
  const minutes = Math.max(1, (secondTs - firstTs) / 60000 || 30);
  const horizon = Math.min(feederRows.length, Math.ceil(Number(byId("horizonSelect").value || 24) * 60 / minutes));
  const rows = feederRows.slice(0, horizon);

  const peak = rows.reduce((b, r) => r.forecast_kw > b.forecast_kw ? r : b, rows[0]);
  const risky = rows.filter(r => r.risk_level !== "normal");
  const avg = rows.reduce((s, r) => s + r.forecast_kw, 0) / rows.length;
  byId("forecastInsight").innerHTML = [
    ["Locality", rows[0].locality],
    ["Peak Forecast", `${peak.forecast_kw.toFixed(2)} kW`],
    ["Peak Window", new Date(peak.timestamp).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"})],
    ["Risk Windows", `${risky.length} intervals`],
  ].map(([l, v]) => `<div class="mini-stat"><span>${l}</span><strong>${v}</strong></div>`).join("");

  const c = chartColors();
  const labels = rows.map(r => new Date(r.timestamp));
  if (state.charts.forecast) state.charts.forecast.destroy();
  state.charts.forecast = new Chart(byId("forecastChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Upper bound", data: rows.map(r => r.upper_kw), borderColor: c.warn, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, fill: false },
        { label: "Forecast", data: rows.map(r => r.forecast_kw), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: "+1", backgroundColor: c.accentFill },
        { label: "Lower bound", data: rows.map(r => r.lower_kw), borderColor: c.warn, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 600 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: true, labels: { color: c.text, font: { family: "Inter", size: 11 }, boxWidth: 14, padding: 12 } },
        tooltip: {
          backgroundColor: isDark() ? "rgba(20,24,32,0.95)" : "rgba(255,255,255,0.95)",
          titleColor: isDark() ? "#e8ecf4" : "#1a1e2e",
          bodyColor: isDark() ? "#8892a4" : "#5a6478",
          borderColor: isDark() ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
          borderWidth: 1,
          padding: 10,
          titleFont: { family: "Inter", weight: "700" },
          bodyFont: { family: "JetBrains Mono", size: 12 },
          callbacks: { title: ctx => new Date(ctx[0].label).toLocaleString(), label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)} kW` },
        },
      },
      scales: {
        x: { type: "timeseries", time: { tooltipFormat: "HH:mm", displayFormats: { hour: "HH:mm" } }, ticks: { color: c.text, font: { size: 10 }, maxTicksLimit: 12 }, grid: { color: c.grid } },
        y: { title: { display: true, text: "kW", color: c.text }, ticks: { color: c.text, font: { family: "JetBrains Mono", size: 10 } }, grid: { color: c.grid } },
      },
    },
  });
}

/* ─── Zones ─── */
function renderZones() {
  byId("zoneMap").innerHTML = state.zones.map(z => `
    <div class="zone-card ${z.feeder_id === state.selectedFeeder ? "active" : ""}" data-feeder="${z.feeder_id}">
      <div>
        <strong>${z.locality}</strong>
        <p>${z.feeder_id} · peak ${Number(z.peak_forecast_kw).toFixed(2)} kW · ${z.open_anomalies} flags</p>
      </div>
      <span class="badge ${z.zone_priority}">${z.zone_priority}</span>
      <div class="bar"><span style="width:${Math.min(100, Number(z.max_risk_score))}%"></span></div>
    </div>
  `).join("");
  document.querySelectorAll(".zone-card").forEach(card => {
    card.addEventListener("click", () => {
      state.selectedFeeder = card.dataset.feeder;
      byId("feederSelect").value = state.selectedFeeder;
      renderForecast();
      renderZones();
      renderAnomalies();
    });
  });
}

/* ─── Anomalies ─── */
function filteredAnomalies() {
  const conf = byId("confidenceFilter").value;
  const search = byId("meterSearch").value.trim().toLowerCase();
  return state.anomalies.filter(a => {
    return (conf === "all" || a.confidence === conf)
      && (!search || String(a.meter_id).toLowerCase().includes(search))
      && (!state.selectedFeeder || a.feeder_id === state.selectedFeeder);
  });
}

function renderAnomalies() {
  const rows = filteredAnomalies();
  byId("anomalyQueue").innerHTML = rows.length ? rows.map(a => `
    <div class="anomaly-card ${a.meter_id === state.selectedMeter ? "active" : ""}" data-meter="${a.meter_id}">
      <div>
        <strong>${a.meter_id}</strong>
        <p>${a.locality} · ${a.segment}</p>
        <p>${a.explanation}</p>
      </div>
      <div>
        <span class="badge ${confidenceClass(a.confidence)}">${a.confidence}</span>
        <p>${Number(a.confidence_score).toFixed(1)}/100</p>
        <p>${inr.format(a.estimated_revenue_risk_inr)}</p>
      </div>
    </div>
  `).join("") : `<div class="evidence-empty">No anomaly flags for current feeder/filter.</div>`;

  document.querySelectorAll(".anomaly-card").forEach(card => {
    card.addEventListener("click", () => { state.selectedMeter = card.dataset.meter; renderAnomalies(); renderEvidence(); });
  });
  if (!state.selectedMeter && rows[0]) { state.selectedMeter = rows[0].meter_id; renderEvidence(); }
  else if (state.selectedMeter) renderEvidence();
}

/* ─── Evidence (Chart.js mini) ─── */
function renderEvidence() {
  const data = state.evidence[state.selectedMeter];
  if (!data) return;
  const meter = data.meter;
  byId("evidenceTitle").textContent = `${meter.meter_id} Evidence`;
  byId("evidenceBadge").className = `badge ${confidenceClass(meter.confidence)}`;
  byId("evidenceBadge").textContent = meter.confidence;
  byId("evidenceBody").className = "evidence-body";

  const chartId = "evidenceMiniChart";
  byId("evidenceBody").innerHTML = `
    <div><h3>Inspection Recommendation</h3><p>${meter.explanation}</p></div>
    <div class="rule-grid">
      ${data.decision_rules.map(r => `<div class="rule ${r.triggered ? "triggered" : ""}"><strong>${r.rule}</strong><p>${r.value} ${r.unit}</p></div>`).join("")}
    </div>
    <div><h3>60-day Evidence Trend</h3><div class="chart-wrap"><canvas id="${chartId}"></canvas></div></div>
  `;

  if (state.charts.evidence) state.charts.evidence.destroy();
  const c = chartColors();
  const pts = data.daily_series;
  state.charts.evidence = new Chart(byId(chartId), {
    type: "line",
    data: {
      labels: pts.map(p => p.date),
      datasets: [
        { label: "Meter", data: pts.map(p => p.daily_kwh), borderColor: c.accent, borderWidth: 2, pointRadius: 0, fill: false },
        { label: "Peer median", data: pts.map(p => p.peer_median_kwh), borderColor: c.warn, borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: c.text, font: { size: 10 } } }, tooltip: { bodyFont: { family: "JetBrains Mono", size: 11 } } },
      scales: {
        x: { ticks: { color: c.text, font: { size: 9 }, maxTicksLimit: 8 }, grid: { color: c.grid } },
        y: { ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
      },
    },
  });
}

/* ─── Pipeline ─── */
function renderPipeline() {
  byId("pipelineStages").innerHTML = state.pipeline.stages.map(s => `<div class="stage"><strong>${s.name}</strong><p>${s.detail}</p></div>`).join("");
  byId("modelCards").innerHTML = state.pipeline.model_cards.map(c => `
    <div class="model-card">
      <h3>${c.name}</h3>
      <p><strong>Model:</strong> ${c.model}</p>
      <p><strong>Target:</strong> ${c.target}</p>
      <p><strong>Features:</strong> ${c.features.join(", ")}</p>
      <p><strong>Output:</strong> ${c.output}</p>
    </div>
  `).join("");
}

/* ─── Theft Validation (Chart.js curves) ─── */
function renderTheftValidation() {
  const tv = state.theftValidation;
  if (!tv || !tv.available) {
    byId("theftMetrics").innerHTML = `<div class="evidence-empty">SGCC validation not available.</div>`;
    byId("featureImportance").innerHTML = "";
    byId("theftCases").innerHTML = "";
    return;
  }
  const metrics = [
    ["Customers", fmt.format(tv.customers)], ["Days", fmt.format(tv.days)],
    ["PR-AUC", tv.pr_auc.toFixed(3)], ["ROC-AUC", tv.roc_auc.toFixed(3)],
    ["Precision", tv.precision.toFixed(3)], ["Recall", tv.recall.toFixed(3)],
    ["F1", tv.f1.toFixed(3)], ["Theft Rate", `${pct.format(tv.positive_rate * 100)}%`],
  ];
  byId("theftMetrics").innerHTML = metrics.map(([l, v]) => `<div class="mini-stat"><span>${l}</span><strong>${v}</strong></div>`).join("");

  const maxImp = Math.max(...tv.feature_importance.map(f => f.importance), 0.001);
  byId("featureImportance").innerHTML = `
    <h3>Top Model Drivers</h3>
    ${tv.feature_importance.map(f => `
      <div class="feature-row">
        <span>${f.feature}</span>
        <div class="bar"><span style="width:${(f.importance / maxImp) * 100}%"></span></div>
        <strong>${f.importance.toFixed(3)}</strong>
      </div>
    `).join("")}
    <div class="eval-grid">
      <div><h3>Confusion Matrix</h3>${renderConfusionMatrix(tv.confusion_matrix)}</div>
      <div><h3>ROC Curve</h3><div class="curve-wrap"><canvas id="rocChart"></canvas></div></div>
      <div><h3>PR Curve</h3><div class="curve-wrap"><canvas id="prChart"></canvas></div></div>
    </div>
  `;

  // Draw ROC and PR curves
  if (state.charts.roc) state.charts.roc.destroy();
  if (state.charts.pr) state.charts.pr.destroy();
  const c = chartColors();
  const curveOpts = (xLab, yLab) => ({
    responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
    plugins: { legend: { display: false }, tooltip: { bodyFont: { family: "JetBrains Mono", size: 11 } } },
    scales: {
      x: { title: { display: true, text: xLab, color: c.text, font: { size: 10 } }, min: 0, max: 1, ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
      y: { title: { display: true, text: yLab, color: c.text, font: { size: 10 } }, min: 0, max: 1, ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
    },
  });
  if (tv.roc_curve?.length) {
    state.charts.roc = new Chart(byId("rocChart"), {
      type: "line",
      data: { datasets: [
        { data: tv.roc_curve.map(p => ({ x: p.fpr, y: p.tpr })), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: false },
        { data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: c.text, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false },
      ] },
      options: curveOpts("False Positive Rate", "True Positive Rate"),
    });
  }
  if (tv.pr_curve?.length) {
    state.charts.pr = new Chart(byId("prChart"), {
      type: "line",
      data: { datasets: [{ data: tv.pr_curve.map(p => ({ x: p.recall, y: p.precision })), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: false }] },
      options: curveOpts("Recall", "Precision"),
    });
  }

  byId("theftCases").innerHTML = tv.top_cases.map(tc => `
    <div class="anomaly-card">
      <div>
        <strong>${tc.consumer_id.slice(0, 10)}…</strong>
        <p>${tc.explanation}</p>
        <p>Ground truth: ${tc.label === 1 ? "⚡ theft" : "✓ normal"}</p>
      </div>
      <div>
        <span class="badge HighConfidence">${pct.format(tc.theft_probability * 100)}%</span>
        <p>${tc.recent_drop_pct}% drop</p>
      </div>
    </div>
  `).join("");
}

function renderConfusionMatrix(cm) {
  if (!cm) return "";
  const [[tn, fp], [fn, tp]] = cm;
  const mx = Math.max(tn, fp, fn, tp, 1);
  const cell = (label, v) => `<div class="cm-cell" style="--alpha:${(0.08 + 0.25 * v / mx).toFixed(2)}"><span>${label}</span><strong>${fmt.format(v)}</strong></div>`;
  return `<div class="confusion">${cell("TN", tn)}${cell("FP", fp)}${cell("FN", fn)}${cell("TP", tp)}</div>`;
}

/* ─── Export ─── */
function exportTheftCsv() {
  const tv = state.theftValidation;
  if (!tv?.available) return;
  const h = ["consumer_id", "label", "theft_probability", "recent_drop_pct", "missing_rate", "zero_rate", "explanation"];
  const rows = tv.top_cases.map(r => h.map(k => `"${String(r[k]).replaceAll('"', '""')}"`).join(","));
  const blob = new Blob([[h.join(","), ...rows].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), { href: url, download: "gridsense_theft_cases.csv" }).click();
  URL.revokeObjectURL(url);
}

/* ─── Init ─── */
async function load() {
  const [metrics, forecasts, zones, anomalies, evidence, pipeline, theftValidation] = await Promise.all([
    getJSON("/api/metrics"), getJSON("/api/forecasts"), getJSON("/api/zones"),
    getJSON("/api/anomalies"), getJSON("/api/anomaly-evidence"),
    getJSON("/api/pipeline"), getJSON("/api/theft-validation"),
  ]);
  Object.assign(state, { metrics, forecasts, zones, anomalies, evidence, pipeline, theftValidation });
  setupControls();
  renderMetrics();
  renderForecast();
  renderZones();
  renderAnomalies();
  renderTheftValidation();
  renderPipeline();
}

initTheme();
byId("darkToggle").addEventListener("click", toggleTheme);
byId("feederSelect").addEventListener("change", e => { state.selectedFeeder = e.target.value; state.selectedMeter = null; renderForecast(); renderZones(); renderAnomalies(); });
byId("horizonSelect").addEventListener("change", renderForecast);
byId("confidenceFilter").addEventListener("change", () => { state.selectedMeter = null; renderAnomalies(); });
byId("meterSearch").addEventListener("input", () => { state.selectedMeter = null; renderAnomalies(); });
byId("refresh").addEventListener("click", load);
byId("exportTheft").addEventListener("click", exportTheftCsv);
byId("rerun").addEventListener("click", async () => {
  byId("rerun").textContent = "Training…";
  byId("rerun").disabled = true;
  await fetch("/api/pipeline/run", { method: "POST" });
  byId("rerun").disabled = false;
  byId("rerun").textContent = "Rebuild Pipeline";
  await load();
});

load().catch(err => {
  document.body.insertAdjacentHTML("afterbegin", `<pre style="margin:16px;padding:16px;background:rgba(239,68,68,0.1);border:1px solid #ef4444;color:#ef4444;border-radius:12px;font-family:var(--mono)">${err.message}</pre>`);
});
