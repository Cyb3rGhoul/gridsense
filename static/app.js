const fmt = new Intl.NumberFormat("en-IN");
const pct = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 });
const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const state = {
  metrics: null,
  forecasts: [],
  zones: [],
  anomalies: [],
  evidence: {},
  pipeline: null,
  theftValidation: null,
  feedback: { records: [], by_meter: {}, summary: {} },
  selectedFeeder: null,
  selectedMeter: null,
  activeView: "overview",
  rebuildPoll: null,
  charts: {},
  map: null,
  mapLayers: {},
};

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} failed with ${res.status}`);
  return res.json();
}

async function getJSONAllowConflict(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${path} failed with ${res.status}`);
  return data;
}

function byId(id) {
  return document.getElementById(id);
}

function confidenceClass(value) {
  return `${String(value || "Low").replace(" ", "")}Confidence`;
}

function priorityColor(priority) {
  return priority === "Critical" ? "#ef4444" : priority === "High" ? "#f59e0b" : "#22c55e";
}

function feedbackSummary() {
  return state.feedback?.summary || {};
}

function meterFeedback(meterId) {
  return state.feedback?.by_meter?.[meterId] || null;
}

function setActiveView(view) {
  state.activeView = view;
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.view === view);
  });
  document.querySelectorAll(".nav-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.viewTarget === view);
  });
  if (view === "risk" && state.map) {
    window.setTimeout(() => state.map.invalidateSize(), 50);
  }
}

function initTheme() {
  const saved = localStorage.getItem("gs-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "");
  updateThemeButton();
}

function updateThemeButton() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  byId("darkToggle").textContent = isLight ? "Light Off" : "Light On";
}

function toggleTheme() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  document.documentElement.setAttribute("data-theme", isLight ? "" : "light");
  localStorage.setItem("gs-theme", isLight ? "dark" : "light");
  updateThemeButton();
  if (state.forecasts.length) renderForecast();
  renderRiskMap();
  renderTheftValidation();
}

function isDark() {
  return document.documentElement.getAttribute("data-theme") !== "light";
}

function chartColors() {
  return {
    text: isDark() ? "#8892a4" : "#5a6478",
    grid: isDark() ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
    accent: "#22d3a7",
    warn: "#f59e0b",
    risk: "#ef4444",
    ok: "#22c55e",
    accentFill: isDark() ? "rgba(34,211,167,0.15)" : "rgba(13,155,122,0.1)",
    tooltipBg: isDark() ? "rgba(20,24,32,0.95)" : "rgba(255,255,255,0.95)",
    tooltipTitle: isDark() ? "#e8ecf4" : "#1a1e2e",
    tooltipBody: isDark() ? "#8892a4" : "#5a6478",
    tooltipBorder: isDark() ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
  };
}

function zoneDisplayPercent(zone) {
  const maxRisk = Math.max(...state.zones.map((z) => Number(z.dispatch_score) || 0), 1);
  const maxAnomalies = Math.max(...state.zones.map((z) => Number(z.open_anomalies) || 0), 1);
  const relativeRisk = ((Number(zone.dispatch_score) || 0) / maxRisk) * 100;
  const anomalyPressure = ((Number(zone.open_anomalies) || 0) / maxAnomalies) * 100;
  const blendedScore = 0.7 * relativeRisk + 0.3 * anomalyPressure;
  const priorityFloor = { Critical: 90, High: 64, Normal: 28 };
  return Math.round(Math.max(priorityFloor[zone.zone_priority] || 20, blendedScore));
}

function uniqueFeeders() {
  return [...new Set(state.forecasts.map((d) => d.feeder_id))].sort();
}

function selectedZone() {
  return state.zones.find((zone) => zone.feeder_id === state.selectedFeeder) || state.zones[0] || null;
}

function reasonBadge(reason) {
  const labels = {
    baseline_drop: "Baseline drop",
    peer_deviation: "Peer deviation",
    persistent_pattern: "Persistent pattern",
  };
  return labels[reason] || String(reason || "").replaceAll("_", " ");
}

function selectFeeder(feederId, targetView = null) {
  state.selectedFeeder = feederId;
  if (byId("feederSelect")) byId("feederSelect").value = feederId;
  state.selectedMeter = null;
  renderForecast();
  renderZones();
  renderRiskMap();
  renderAnomalies();
  renderOperatorStatus();
  if (targetView) setActiveView(targetView);
}

function renderMetrics() {
  const m = state.metrics;
  const smape = m.forecast_smape || m.forecast_mape || 0;
  const criticalZones = state.zones.filter((zone) => zone.zone_priority === "Critical").length;
  const dispatchCases = state.anomalies.filter((anomaly) => anomaly.inspection_priority === "Dispatch").length;
  const feedback = feedbackSummary();
  const guardrail = m.anomaly_precision_target
    ? `Precision ${pct.format(m.anomaly_precision_target * 100)}%`
    : "Persistence gated";
  const cards = [
    ["Forecast Feed", "Official/Public", m.forecast_dataset_source || m.dataset_source],
    ["Forecast sMAPE", `${pct.format(smape * 100)}%`, `Baseline ${pct.format((m.forecast_baseline_smape || 0) * 100)}%`],
    ["Critical Zones", `${criticalZones}`, `${state.zones.length} mapped localities under watch`],
    ["Dispatch Cases", `${dispatchCases}`, `Queue rows ${fmt.format(state.anomalies.length)}`],
    ["Queue Mode", m.anomaly_mode?.includes("supervised") ? "Labelled" : "Operational", m.anomaly_mode],
    [
      "Field Feedback",
      `${feedback.confirmed_suspicious || 0}/${feedback.false_alarm || 0}`,
      `Confirmed vs false alarms${feedback.total ? ` across ${feedback.total} reviewed cases` : " waiting for field review"}`,
    ],
    ["Queue Guardrail", guardrail, m.anomaly_threshold_policy],
  ];
  byId("metrics").innerHTML = cards
    .map(
      ([label, value, hint]) => `
      <article class="card metric-card">
        <div class="metric-head">
          <span class="metric-label">${label}</span>
        </div>
        <b class="metric-value">${value}</b>
        <span class="hint metric-hint">${hint}</span>
      </article>
    `,
    )
    .join("");
  byId("lastUpdated").textContent = new Date(m.generated_at).toLocaleString();
}

function setRerunButton(status, startedAt = null) {
  const button = byId("rerun");
  if (status === "running") {
    button.disabled = true;
    button.textContent = startedAt
      ? `Rebuilding since ${new Date(startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
      : "Rebuilding...";
    return;
  }
  button.disabled = false;
  button.textContent = "Rebuild Pipeline";
}

async function syncPipelineStatus() {
  const status = await getJSON("/api/pipeline/status");
  setRerunButton(status.status, status.started_at);
  if (status.status !== "running") {
    if (state.rebuildPoll) {
      window.clearInterval(state.rebuildPoll);
      state.rebuildPoll = null;
    }
    return status;
  }
  if (!state.rebuildPoll) {
    state.rebuildPoll = window.setInterval(async () => {
      const next = await getJSON("/api/pipeline/status");
      setRerunButton(next.status, next.started_at);
      if (["completed", "failed", "idle"].includes(next.status)) {
        window.clearInterval(state.rebuildPoll);
        state.rebuildPoll = null;
        if (next.status === "completed") {
          await load();
        } else if (next.status === "failed") {
          document.body.insertAdjacentHTML(
            "afterbegin",
            `<pre style="margin:16px;padding:16px;background:rgba(239,68,68,0.1);border:1px solid #ef4444;color:#ef4444;border-radius:12px;font-family:var(--mono)">${next.last_error || "Pipeline rebuild failed"}</pre>`,
          );
        }
      }
    }, 2500);
  }
  return status;
}

function renderProvenance() {
  const m = state.metrics;
  const tv = state.theftValidation;
  const feedback = feedbackSummary();
  const cards = [
    {
      kicker: "Operational Feed",
      title: "Forecasting and anomaly triage stay separated so each job uses the right dataset and model.",
      body: `Forecasts use ${m.forecast_dataset_source}. Anomaly triage uses ${m.anomaly_dataset_source}. This keeps feeder stress tracking and inspection scoring honest instead of blending incompatible sources.`,
    },
    {
      kicker: "Queue Logic",
      title: "Cases move into the queue only after evidence support and false-positive controls agree.",
      body: m.anomaly_false_positive_controls || "Operational queue uses explainable controls before recommending inspection.",
    },
    {
      kicker: "Validation Lab",
      title: "SGCC remains a separate labelled benchmark and never drives the live queue.",
      body: tv?.available
        ? "Use the Validation Lab for benchmark metrics and model evidence only. The operational dashboard remains a separate decision-support layer."
        : "Labelled benchmark data is not available in this build.",
    },
    {
      kicker: "Field Feedback",
      title: "Inspection outcomes are persisted so false alarms stay visible and thresholds can improve over time.",
      body: feedback.total
        ? `${feedback.confirmed_suspicious || 0} suspicious confirmations, ${feedback.false_alarm || 0} false alarms, and ${feedback.cleared || 0} cleared cases have been logged in this environment.`
        : "No field outcomes have been logged yet. Review actions from the evidence panel feed the audit trail.",
    },
  ];
  byId("provenance").innerHTML = cards
    .map(
      (card) => `
      <article class="provenance-card">
        <span>${card.kicker}</span>
        <h3>${card.title}</h3>
        <p>${card.body}</p>
      </article>
    `,
    )
    .join("");
}

function renderSituationSummary() {
  const topZone = [...state.zones].sort((a, b) => Number(b.dispatch_score || 0) - Number(a.dispatch_score || 0))[0];
  const dispatchCases = state.anomalies.filter((a) => a.inspection_priority === "Dispatch").length;
  const inspectCases = state.anomalies.filter((a) => a.inspection_priority === "Inspect").length;
  const feedback = feedbackSummary();
  const summary = [
    {
      label: "Most Exposed Zone",
      value: topZone ? topZone.locality : "Waiting",
      hint: topZone ? `Dispatch score ${Number(topZone.dispatch_score).toFixed(1)}. ${topZone.recommended_action}.` : "No zones available.",
    },
    {
      label: "Immediate Cases",
      value: `${dispatchCases}`,
      hint: `${inspectCases} additional cases are queued for targeted meter audit.`,
    },
    {
      label: "Forecast Lift",
      value: `${pct.format(((state.metrics.forecast_baseline_smape || 0) - (state.metrics.forecast_smape || 0)) * 100)} pts`,
      hint: "Improvement over the 24-hour persistence baseline.",
    },
    {
      label: "Queue Gate",
      value: state.metrics.anomaly_mode?.includes("supervised") ? "Precision-led" : "Evidence-led",
      hint: state.metrics.anomaly_threshold_policy,
    },
    {
      label: "Feedback Loop",
      value: feedback.total ? `${feedback.total} reviews` : "Awaiting reviews",
      hint: feedback.total
        ? `${feedback.confirmed_suspicious || 0} confirmed suspicious and ${feedback.false_alarm || 0} false alarms recorded.`
        : "Use the evidence panel to mark suspicious, false alarm, or cleared outcomes.",
    },
  ];
  byId("situationSummary").innerHTML = summary
    .map(
      (item) => `
      <div class="summary-stat">
        <span>${item.label}</span>
        <strong>${item.value}</strong>
        <p>${item.hint}</p>
      </div>
    `,
    )
    .join("");
}

function renderOperatorStatus() {
  const zone = selectedZone();
  const feedback = feedbackSummary();
  const blocks = [
    ["Forecast Feed", state.metrics.forecast_dataset_source || state.metrics.dataset_source],
    ["Queue Gate", state.metrics.anomaly_threshold_policy || "Threshold policy not available."],
    ["False Positive Controls", state.metrics.anomaly_false_positive_controls || state.metrics.visible_false_positive_proxy],
    ["Field Review", feedback.total ? `${feedback.confirmed_suspicious || 0} confirmed | ${feedback.false_alarm || 0} false alarms` : "No field outcomes recorded yet"],
    ["Current Focus", zone ? `${zone.locality}: ${zone.recommended_action}` : "No selected feeder"],
  ];
  byId("operatorStatus").innerHTML = blocks
    .map(
      ([label, value]) => `
      <div class="summary-stat">
        <span>${label}</span>
        <p>${value}</p>
      </div>
    `,
    )
    .join("");
}

function setupControls() {
  const feeders = uniqueFeeders();
  if (!state.selectedFeeder || !feeders.includes(state.selectedFeeder)) state.selectedFeeder = feeders[0];
  byId("feederSelect").innerHTML = feeders.map((id) => `<option value="${id}">${id}</option>`).join("");
  byId("feederSelect").value = state.selectedFeeder;
}

function renderForecast() {
  const feederRows = state.forecasts.filter((d) => d.feeder_id === state.selectedFeeder);
  if (!feederRows.length) return;
  const firstTs = new Date(feederRows[0].timestamp);
  const secondTs = new Date(feederRows[1]?.timestamp || firstTs);
  const minutes = Math.max(1, (secondTs - firstTs) / 60000 || 30);
  const horizon = Math.min(feederRows.length, Math.ceil((Number(byId("horizonSelect").value || 24) * 60) / minutes));
  const rows = feederRows.slice(0, horizon);
  const peak = rows.reduce((best, row) => (row.forecast_kw > best.forecast_kw ? row : best), rows[0]);
  const zone = selectedZone();

  byId("forecastInsight").innerHTML = [
    ["Locality", rows[0].locality],
    ["Peak Forecast", `${peak.forecast_kw.toFixed(2)} kW`],
    ["Peak Window", new Date(peak.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })],
    ["Utilization", `${pct.format(Number(zone?.peak_utilization_pct || peak.capacity_utilization_pct || 0))}%`],
  ]
    .map(([label, value]) => `<div class="mini-stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  const c = chartColors();
  const labels = rows.map((row) => new Date(row.timestamp));
  if (state.charts.forecast) state.charts.forecast.destroy();
  state.charts.forecast = new Chart(byId("forecastChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Upper bound", data: rows.map((r) => r.upper_kw), borderColor: c.warn, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, fill: false },
        { label: "Forecast", data: rows.map((r) => r.forecast_kw), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: "+1", backgroundColor: c.accentFill },
        { label: "Lower bound", data: rows.map((r) => r.lower_kw), borderColor: c.warn, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: true, labels: { color: c.text, font: { size: 11 }, boxWidth: 14, padding: 12 } },
        tooltip: {
          backgroundColor: c.tooltipBg,
          titleColor: c.tooltipTitle,
          bodyColor: c.tooltipBody,
          borderColor: c.tooltipBorder,
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: (ctx) => new Date(ctx[0].label).toLocaleString(),
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)} kW`,
          },
        },
      },
      scales: {
        x: { type: "timeseries", time: { tooltipFormat: "HH:mm", displayFormats: { hour: "HH:mm" } }, ticks: { color: c.text, font: { size: 10 }, maxTicksLimit: 12 }, grid: { color: c.grid } },
        y: { title: { display: true, text: "kW", color: c.text }, ticks: { color: c.text, font: { size: 10 } }, grid: { color: c.grid } },
      },
    },
  });
}

function renderZoneCard(zone, compact = false) {
  const active = zone.feeder_id === state.selectedFeeder ? "active" : "";
  const percent = zoneDisplayPercent(zone);
  const meta = compact
    ? `${zone.zone_priority} zone | Dispatch ${Number(zone.dispatch_score).toFixed(1)}`
    : `${zone.feeder_id} | ${Number(zone.peak_forecast_kw).toFixed(1)} kW peak | ${zone.open_anomalies} queue cases`;
  const details = compact
    ? `${zone.recommended_action}`
    : `${pct.format(Number(zone.peak_utilization_pct || 0))}% utilization | ${inr.format(Number(zone.revenue_risk_inr || 0))} exposure | ${zone.recommended_action}`;
  return `
    <div class="zone-card ${active}" data-feeder="${zone.feeder_id}">
      <div class="zone-main">
        <div class="title-row">
          <strong>${zone.locality}</strong>
          <span class="metric-pill">${Number(zone.dispatch_score).toFixed(1)}</span>
        </div>
        <p>${meta}</p>
        <div class="zone-meta">
          <span>${details}</span>
        </div>
      </div>
      <div class="zone-side">
        <span class="badge ${zone.zone_priority}">${zone.zone_priority}</span>
      </div>
      <div class="bar"><span style="width:${percent}%"></span></div>
    </div>
  `;
}

function bindZoneClicks() {
  document.querySelectorAll(".zone-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectFeeder(card.dataset.feeder, "risk");
    });
  });
}

function renderZones() {
  const ordered = [...state.zones].sort((a, b) => Number(b.dispatch_score || 0) - Number(a.dispatch_score || 0));
  byId("zoneMap").innerHTML = ordered.map((zone) => renderZoneCard(zone, false)).join("");
  byId("overviewZones").innerHTML = ordered.slice(0, 4).map((zone) => renderZoneCard(zone, true)).join("");
  bindZoneClicks();
}

function filteredAnomalies() {
  const conf = byId("confidenceFilter").value;
  const search = byId("meterSearch").value.trim().toLowerCase();
  return state.anomalies.filter((a) => {
    return (conf === "all" || a.confidence === conf)
      && (!search || String(a.meter_id).toLowerCase().includes(search))
      && (!state.selectedFeeder || a.feeder_id === state.selectedFeeder);
  });
}

function renderRiskMap() {
  const mapTarget = byId("riskGeoMap");
  const mappableZones = state.zones.filter((zone) => Number.isFinite(Number(zone.lat)) && Number.isFinite(Number(zone.lon)));
  if (!mapTarget) return;
  if (!window.L || !mappableZones.length) {
    mapTarget.innerHTML = `<div class="evidence-empty">Real locality coordinates are not available for the current data build.</div>`;
    return;
  }
  if (!state.map) {
    mapTarget.innerHTML = `<div id="riskLeafletMap" style="width:100%;height:100%;min-height:520px"></div>
      <div class="map-legend">
        <span>Dispatch Legend</span>
        <div class="legend-row"><i class="legend-swatch" style="background:#ef4444"></i> Critical zone</div>
        <div class="legend-row"><i class="legend-swatch" style="background:#f59e0b"></i> High zone</div>
        <div class="legend-row"><i class="legend-swatch" style="background:#22c55e"></i> Normal zone</div>
      </div>`;
    state.map = L.map("riskLeafletMap", { zoomControl: true, scrollWheelZoom: true, preferCanvas: true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(state.map);
    state.mapLayers = {
      zones: L.layerGroup().addTo(state.map),
      anomalies: L.layerGroup().addTo(state.map),
    };
    L.control.layers({}, { "Zone dispatch": state.mapLayers.zones, "Anomaly exposure": state.mapLayers.anomalies }, { collapsed: false }).addTo(state.map);
  }

  state.mapLayers.zones.clearLayers();
  state.mapLayers.anomalies.clearLayers();

  const bounds = [];
  const selected = selectedZone();
  mappableZones.forEach((zone) => {
    const lat = Number(zone.lat);
    const lon = Number(zone.lon);
    const color = priorityColor(zone.zone_priority);
    const isSelected = zone.feeder_id === state.selectedFeeder;
    const areaRadius = 250 + Number(zone.dispatch_score || 0) * 18 + Number(zone.open_anomalies || 0) * 40;
    const ring = L.circle([lat, lon], {
      color,
      weight: isSelected ? 3 : 2,
      opacity: 0.72,
      fillColor: color,
      fillOpacity: isSelected ? 0.22 : 0.12,
      radius: areaRadius,
    });
    const marker = L.circleMarker([lat, lon], {
      radius: isSelected ? 12 : 9,
      color: "#ffffff",
      weight: isSelected ? 2.5 : 1.5,
      fillColor: color,
      fillOpacity: 0.96,
    });
    const popup = `
      <div class="map-popup">
        <strong>${zone.locality}</strong>
        <p>${zone.locality_display_name || zone.locality}</p>
        <p>${zone.zone_priority} | Dispatch ${Number(zone.dispatch_score).toFixed(1)} | Peak utilization ${pct.format(Number(zone.peak_utilization_pct || 0))}%</p>
        <p>${Number(zone.open_anomalies || 0)} live anomaly cases | ${inr.format(Number(zone.revenue_risk_inr || 0))} estimated exposure</p>
        <p>${zone.recommended_action}</p>
      </div>
    `;
    ring.bindPopup(popup);
    marker.bindPopup(popup);
    ring.on("click", () => selectFeeder(zone.feeder_id, "risk"));
    marker.on("click", () => selectFeeder(zone.feeder_id, "risk"));
    ring.addTo(state.mapLayers.zones);
    marker.addTo(state.mapLayers.zones);
    bounds.push([lat, lon]);
  });

  filteredAnomalies()
    .filter((item) => Number.isFinite(Number(item.lat)) && Number.isFinite(Number(item.lon)))
    .forEach((anomaly) => {
      const marker = L.circleMarker([Number(anomaly.lat), Number(anomaly.lon)], {
        radius: 4 + Number(anomaly.confidence_score || 0) / 18,
        color: "#111827",
        weight: 1.25,
        fillColor: anomaly.inspection_priority === "Dispatch" ? "#ef4444" : anomaly.inspection_priority === "Inspect" ? "#f59e0b" : "#60a5fa",
        fillOpacity: 0.9,
      });
      marker.bindPopup(`
        <div class="map-popup">
          <strong>${anomaly.meter_id}</strong>
          <p>${anomaly.locality} | ${anomaly.suspicion_type.replaceAll("_", " ")}</p>
          <p>${anomaly.confidence} confidence ${Number(anomaly.confidence_score).toFixed(0)}/100 | FP risk ${Number(anomaly.false_positive_risk || 0).toFixed(0)}</p>
          <p>${anomaly.recommended_action}</p>
        </div>
      `);
      marker.on("click", () => {
        state.selectedMeter = anomaly.meter_id;
        state.selectedFeeder = anomaly.feeder_id;
        if (byId("feederSelect")) byId("feederSelect").value = anomaly.feeder_id;
        renderForecast();
        renderZones();
        renderAnomalies();
        renderOperatorStatus();
        setActiveView("queue");
      });
      marker.addTo(state.mapLayers.anomalies);
    });

  if (bounds.length) state.map.fitBounds(bounds, { padding: [24, 24], maxZoom: 12 });
  if (selected && Number.isFinite(Number(selected.lat)) && Number.isFinite(Number(selected.lon))) {
    state.map.panTo([Number(selected.lat), Number(selected.lon)], { animate: true });
  }
}

function renderAnomalyCard(anomaly, compact = false) {
  const active = anomaly.meter_id === state.selectedMeter ? "active" : "";
  const reasons = (anomaly.reason_codes || []).map((code) => `<span class="signal-tag">${reasonBadge(code)}</span>`).join("");
  const secondary = compact
    ? `${anomaly.locality} | ${anomaly.inspection_priority} | ${inr.format(Number(anomaly.estimated_revenue_risk_inr || 0))}`
    : `${anomaly.locality} | ${anomaly.segment} | ${anomaly.suspicion_type.replaceAll("_", " ")}`;
  const controls = compact
    ? `${Number(anomaly.confidence_score).toFixed(0)}/100 confidence`
    : `FP risk ${Number(anomaly.false_positive_risk || 0).toFixed(0)} | ${anomaly.stability_days} stable days | ${Number(anomaly.confidence_score || 0).toFixed(0)}/100`;
  return `
    <div class="anomaly-card ${active}" data-meter="${anomaly.meter_id}">
      <div class="anomaly-main">
        <div class="title-row">
          <strong>${anomaly.meter_id}</strong>
          <span class="metric-pill">${inr.format(Number(anomaly.estimated_revenue_risk_inr || 0))}</span>
        </div>
        <p>${secondary}</p>
        <p>${anomaly.explanation}</p>
        <div class="anomaly-meta">
          <span class="signal-tag action-tag">${anomaly.recommended_action}</span>
          <span class="signal-tag">${controls}</span>
          ${reasons}
        </div>
      </div>
      <div class="anomaly-side">
        <span class="badge ${confidenceClass(anomaly.confidence)}">${anomaly.confidence}</span>
        <strong class="side-stat">${anomaly.inspection_priority}</strong>
        <span class="side-note">${anomaly.locality}</span>
      </div>
    </div>
  `;
}

function bindAnomalyClicks() {
  document.querySelectorAll(".anomaly-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedMeter = card.dataset.meter;
      renderAnomalies();
      renderEvidence();
      setActiveView("queue");
    });
  });
}

function renderAnomalies() {
  const rows = filteredAnomalies();
  byId("anomalyQueue").innerHTML = rows.length
    ? rows.map((anomaly) => renderAnomalyCard(anomaly, false)).join("")
    : `<div class="evidence-empty">No anomaly flags for the current feeder and filter.</div>`;

  byId("overviewQueue").innerHTML = rows.length
    ? rows.slice(0, 4).map((anomaly) => renderAnomalyCard(anomaly, true)).join("")
    : `<div class="evidence-empty">No high-priority operational cases are currently active.</div>`;

  bindAnomalyClicks();
  if (!state.selectedMeter && rows[0]) state.selectedMeter = rows[0].meter_id;
  renderEvidence();
}

async function submitFeedback(verdict) {
  const data = state.evidence[state.selectedMeter];
  if (!data?.meter) return;
  const noteInput = byId("feedbackNote");
  await getJSONAllowConflict("/api/inspection-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      meter_id: data.meter.meter_id,
      feeder_id: data.meter.feeder_id,
      locality: data.meter.locality,
      verdict,
      note: noteInput ? noteInput.value.trim() : "",
    }),
  });
  state.feedback = await getJSON("/api/inspection-feedback");
  renderMetrics();
  renderProvenance();
  renderSituationSummary();
  renderOperatorStatus();
  renderEvidence();
}

function bindFeedbackActions() {
  document.querySelectorAll("[data-feedback-verdict]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await submitFeedback(button.dataset.feedbackVerdict);
      } finally {
        button.disabled = false;
      }
    });
  });
}

function renderEvidence() {
  const data = state.evidence[state.selectedMeter];
  if (!data) {
    byId("evidenceTitle").textContent = "Select an operational anomaly";
    byId("evidenceBadge").className = "badge";
    byId("evidenceBadge").textContent = "Waiting";
    byId("evidenceBody").className = "evidence-empty";
    byId("evidenceBody").innerHTML = "Pick a meter from the operational anomaly queue to see evidence and guidance.";
    return;
  }

  const meter = data.meter;
  const review = meterFeedback(meter.meter_id);
  byId("evidenceTitle").textContent = `${meter.meter_id} Operational Evidence`;
  byId("evidenceBadge").className = `badge ${confidenceClass(meter.confidence)}`;
  byId("evidenceBadge").textContent = meter.confidence;
  byId("evidenceBody").className = "evidence-body";

  const chartId = "evidenceMiniChart";
  byId("evidenceBody").innerHTML = `
    <div class="summary-grid">
      <div class="summary-stat">
        <span>Recommended action</span>
        <strong>${meter.recommended_action}</strong>
        <p>${meter.inspection_priority} priority with false-positive risk ${Number(meter.false_positive_risk || 0).toFixed(0)}.</p>
      </div>
      <div class="summary-stat">
        <span>Suspicion pattern</span>
        <strong>${meter.suspicion_type.replaceAll("_", " ")}</strong>
        <p>${meter.stability_days} stable days across recent observation windows.</p>
      </div>
      <div class="summary-stat">
        <span>Latest field review</span>
        <strong>${review ? review.verdict.replaceAll("_", " ") : "Not reviewed"}</strong>
        <p>${review ? `Recorded ${new Date(review.recorded_at).toLocaleString()}` : "Persisted review outcomes tighten the false-positive story over time."}</p>
      </div>
    </div>
    <div>
      <h3>Inspection reasoning</h3>
      <p>${meter.explanation}</p>
    </div>
    <div class="feedback-panel">
      <div class="feedback-actions">
        <button type="button" class="ghost feedback-btn" data-feedback-verdict="confirmed_suspicious">Confirm Suspicious</button>
        <button type="button" class="ghost feedback-btn" data-feedback-verdict="false_alarm">Mark False Alarm</button>
        <button type="button" class="ghost feedback-btn" data-feedback-verdict="cleared">Mark Cleared</button>
      </div>
      <label class="feedback-note-label">
        Review note
        <input id="feedbackNote" placeholder="Optional field note for audit trail" value="${review?.note || ""}" />
      </label>
    </div>
    <div class="rule-grid">
      ${data.decision_rules.map((rule) => `<div class="rule ${rule.triggered ? "triggered" : ""}"><strong>${rule.rule}</strong><p>${rule.value} ${rule.unit}</p></div>`).join("")}
    </div>
    <div>
      <h3>60-day evidence trend</h3>
      <div class="chart-wrap"><canvas id="${chartId}"></canvas></div>
    </div>
  `;

  bindFeedbackActions();

  if (state.charts.evidence) state.charts.evidence.destroy();
  const c = chartColors();
  const points = data.daily_series;
  state.charts.evidence = new Chart(byId(chartId), {
    type: "line",
    data: {
      labels: points.map((point) => point.date),
      datasets: [
        { label: "Meter", data: points.map((point) => point.daily_kwh), borderColor: c.accent, borderWidth: 2, pointRadius: 0, fill: false },
        { label: "Peer median", data: points.map((point) => point.peer_median_kwh), borderColor: c.warn, borderWidth: 1.5, borderDash: [4, 3], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: c.text, font: { size: 10 } } },
        tooltip: { bodyFont: { size: 11 } },
      },
      scales: {
        x: { ticks: { color: c.text, font: { size: 9 }, maxTicksLimit: 8 }, grid: { color: c.grid } },
        y: { ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
      },
    },
  });
}

function renderPipeline() {
  byId("pipelineStages").innerHTML = state.pipeline.stages
    .map((stage) => `<div class="stage"><strong>${stage.name}</strong><p>${stage.detail}</p></div>`)
    .join("");
  byId("modelCards").innerHTML = state.pipeline.model_cards
    .map(
      (card) => `
      <div class="model-card">
        <h3>${card.name}</h3>
        <p><strong>Model:</strong> ${card.model}</p>
        <p><strong>Target:</strong> ${card.target}</p>
        <p><strong>Features:</strong> ${card.features.join(", ")}</p>
        <p><strong>Output:</strong> ${card.output}</p>
      </div>
    `,
    )
    .join("");
}

function renderConfusionMatrix(cm) {
  if (!cm) return "";
  const [[tn, fp], [fn, tp]] = cm;
  const maxValue = Math.max(tn, fp, fn, tp, 1);
  const cell = (label, value) => `<div class="cm-cell" style="--alpha:${(0.08 + (0.25 * value) / maxValue).toFixed(2)}"><span>${label}</span><strong>${fmt.format(value)}</strong></div>`;
  return `<div class="confusion">${cell("TN", tn)}${cell("FP", fp)}${cell("FN", fn)}${cell("TP", tp)}</div>`;
}

function renderTheftValidation() {
  const tv = state.theftValidation;
  if (!tv || !tv.available) {
    byId("theftMetrics").innerHTML = `<div class="evidence-empty">SGCC validation is not available.</div>`;
    byId("featureImportance").innerHTML = "";
    byId("theftCases").innerHTML = "";
    return;
  }

  const metrics = [
    ["Customers", fmt.format(tv.customers)],
    ["Days", fmt.format(tv.days)],
    ["PR-AUC", tv.pr_auc.toFixed(3)],
    ["ROC-AUC", tv.roc_auc.toFixed(3)],
    ["Precision", tv.precision.toFixed(3)],
    ["Recall", tv.recall.toFixed(3)],
    ["F1", tv.f1.toFixed(3)],
    ["Theft Rate", `${pct.format(tv.positive_rate * 100)}%`],
  ];
  byId("theftMetrics").innerHTML = metrics
    .map(([label, value]) => `<div class="mini-stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  const maxImportance = Math.max(...tv.feature_importance.map((feature) => feature.importance), 0.001);
  byId("featureImportance").innerHTML = `
    <p class="section-note">This SGCC block is a separate labelled benchmark. It validates theft detection quality but does not drive the live operational queue. The live queue currently uses a precision-biased threshold at probability >= ${state.metrics.anomaly_queue_probability_threshold ? state.metrics.anomaly_queue_probability_threshold.toFixed(2) : "N/A"}.</p>
    <h3>Top model drivers</h3>
    ${tv.feature_importance
      .map(
        (feature) => `
        <div class="feature-row">
          <span>${feature.feature}</span>
          <div class="bar"><span style="width:${(feature.importance / maxImportance) * 100}%"></span></div>
          <strong>${feature.importance.toFixed(3)}</strong>
        </div>
      `,
      )
      .join("")}
    <div class="eval-grid">
      <div><h3>Confusion matrix</h3>${renderConfusionMatrix(tv.confusion_matrix)}</div>
      <div><h3>ROC curve</h3><div class="curve-wrap"><canvas id="rocChart"></canvas></div></div>
      <div><h3>PR curve</h3><div class="curve-wrap"><canvas id="prChart"></canvas></div></div>
    </div>
  `;

  if (state.charts.roc) state.charts.roc.destroy();
  if (state.charts.pr) state.charts.pr.destroy();
  const c = chartColors();
  const curveOptions = (xLabel, yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 500 },
    plugins: { legend: { display: false }, tooltip: { bodyFont: { size: 11 } } },
    scales: {
      x: { title: { display: true, text: xLabel, color: c.text, font: { size: 10 } }, min: 0, max: 1, ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
      y: { title: { display: true, text: yLabel, color: c.text, font: { size: 10 } }, min: 0, max: 1, ticks: { color: c.text, font: { size: 9 } }, grid: { color: c.grid } },
    },
  });

  if (tv.roc_curve?.length) {
    state.charts.roc = new Chart(byId("rocChart"), {
      type: "line",
      data: {
        datasets: [
          { data: tv.roc_curve.map((point) => ({ x: point.fpr, y: point.tpr })), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: false },
          { data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: c.text, borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false },
        ],
      },
      options: curveOptions("False Positive Rate", "True Positive Rate"),
    });
  }

  if (tv.pr_curve?.length) {
    state.charts.pr = new Chart(byId("prChart"), {
      type: "line",
      data: {
        datasets: [{ data: tv.pr_curve.map((point) => ({ x: point.recall, y: point.precision })), borderColor: c.accent, borderWidth: 2.5, pointRadius: 0, fill: false }],
      },
      options: curveOptions("Recall", "Precision"),
    });
  }

  byId("theftCases").innerHTML = tv.top_cases
    .map(
      (tc) => `
      <div class="anomaly-card">
        <div>
          <strong>${tc.consumer_id.slice(0, 10)}...</strong>
          <p>${tc.explanation}</p>
          <p>Labelled validation case: ${tc.label === 1 ? "theft" : "normal"}</p>
        </div>
        <div>
          <span class="badge HighConfidence">${pct.format(tc.theft_probability * 100)}%</span>
          <p>${tc.recent_drop_pct}% drop</p>
        </div>
      </div>
    `,
    )
    .join("");
}

function exportTheftCsv() {
  const tv = state.theftValidation;
  if (!tv?.available) return;
  const headers = ["consumer_id", "label", "theft_probability", "recent_drop_pct", "missing_rate", "zero_rate", "explanation"];
  const rows = tv.top_cases.map((row) => headers.map((key) => `"${String(row[key]).replaceAll('"', '""')}"`).join(","));
  const blob = new Blob([[headers.join(","), ...rows].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), { href: url, download: "gridsense_theft_cases.csv" }).click();
  URL.revokeObjectURL(url);
}

async function load() {
  const [metrics, forecasts, zones, anomalies, evidence, pipeline, theftValidation, feedback] = await Promise.all([
    getJSON("/api/metrics"),
    getJSON("/api/forecasts"),
    getJSON("/api/zones"),
    getJSON("/api/anomalies"),
    getJSON("/api/anomaly-evidence"),
    getJSON("/api/pipeline"),
    getJSON("/api/theft-validation"),
    getJSON("/api/inspection-feedback"),
  ]);

  Object.assign(state, { metrics, forecasts, zones, anomalies, evidence, pipeline, theftValidation, feedback });
  setupControls();
  renderMetrics();
  renderProvenance();
  renderSituationSummary();
  renderForecast();
  renderZones();
  renderRiskMap();
  renderAnomalies();
  renderOperatorStatus();
  renderTheftValidation();
  renderPipeline();
  setActiveView(state.activeView);
  await syncPipelineStatus();
}

initTheme();
document.querySelectorAll(".nav-chip").forEach((chip) => {
  chip.addEventListener("click", () => setActiveView(chip.dataset.viewTarget));
});
byId("darkToggle").addEventListener("click", toggleTheme);
byId("feederSelect").addEventListener("change", (event) => {
  selectFeeder(event.target.value);
});
byId("horizonSelect").addEventListener("change", renderForecast);
byId("confidenceFilter").addEventListener("change", () => {
  state.selectedMeter = null;
  renderAnomalies();
});
byId("meterSearch").addEventListener("input", () => {
  state.selectedMeter = null;
  renderAnomalies();
});
byId("refresh").addEventListener("click", load);
byId("exportTheft").addEventListener("click", exportTheftCsv);
byId("rerun").addEventListener("click", async () => {
  const run = await getJSONAllowConflict("/api/pipeline/run", { method: "POST" });
  setRerunButton("running", run.started_at);
  await syncPipelineStatus();
});

load().catch((err) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<pre style="margin:16px;padding:16px;background:rgba(239,68,68,0.1);border:1px solid #ef4444;color:#ef4444;border-radius:12px;font-family:var(--mono)">${err.message}</pre>`,
  );
});
