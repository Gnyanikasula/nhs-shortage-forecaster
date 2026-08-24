// All requests are same-origin -- this page is served BY the FastAPI app
// itself (mounted as static files), so relative paths hit the same
// service's API routes directly. No CORS configuration needed, no
// separate API_BASE_URL to manage across environments.

let latestPredictions = [];

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`${url} returned HTTP ${resp.status}`);
  }
  return resp.json();
}

function showError(message) {
  const banner = document.createElement("div");
  banner.className = "error-banner";
  banner.textContent = message;
  document.querySelector("main").prepend(banner);
}

async function loadStats() {
  const stats = await fetchJSON("/stats");
  document.getElementById("stat-outcomes").textContent = stats.total_outcome_rows.toLocaleString();
  document.getElementById("stat-predictions").textContent = stats.total_prediction_rows.toLocaleString();
  document.getElementById("stat-month").textContent = stats.latest_prediction_month || "--";
  return stats;
}

async function loadLatestPredictions() {
  const data = await fetchJSON("/predictions/latest?limit=377");
  latestPredictions = data.predictions;
  document.getElementById("risk-heading").textContent = `Top risk chemicals -- ${data.month || "no data"}`;
  return data;
}

let riskChart = null;

function renderRiskTableAndChart(topN) {
  const top = latestPredictions.slice(0, topN);

  const tbody = document.getElementById("risk-table-body");
  tbody.innerHTML = "";
  top.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.chemical}</td><td class="score">${row.phase1_production_score.toFixed(4)}</td>`;
    tbody.appendChild(tr);
  });

  const ctx = document.getElementById("risk-chart").getContext("2d");
  const chartData = {
    labels: top.map((r) => r.chemical),
    datasets: [{
      label: "Onset risk (next month)",
      data: top.map((r) => r.phase1_production_score),
      backgroundColor: "#7cc4ff",
    }],
  };

  if (riskChart) {
    riskChart.data = chartData;
    riskChart.update();
  } else {
    riskChart = new Chart(ctx, {
      type: "bar",
      data: chartData,
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9aa4b2", maxRotation: 90, minRotation: 45 } },
          y: { ticks: { color: "#9aa4b2" }, beginAtZero: true },
        },
      },
    });
  }
}

function populateChemicalSelect() {
  const select = document.getElementById("chemical-select");
  select.innerHTML = "";
  latestPredictions.forEach((row) => {
    const opt = document.createElement("option");
    opt.value = row.chemical;
    opt.textContent = row.chemical;
    select.appendChild(opt);
  });
}

let historyChart = null;

async function loadChemicalHistory(chemical) {
  let data;
  try {
    data = await fetchJSON(`/outcomes/${encodeURIComponent(chemical)}?months=79`);
  } catch (e) {
    document.getElementById("history-chart").getContext("2d").clearRect(0, 0, 9999, 9999);
    return;
  }

  const history = data.history;
  const ctx = document.getElementById("history-chart").getContext("2d");
  const chartData = {
    labels: history.map((h) => h.month),
    datasets: [{
      label: "On concession (1 = yes)",
      data: history.map((h) => (h.on_concession ? 1 : 0)),
      borderColor: "#4a9eff",
      backgroundColor: "rgba(74, 158, 255, 0.15)",
      stepped: true,
      fill: true,
    }],
  };

  if (historyChart) {
    historyChart.data = chartData;
    historyChart.update();
  } else {
    historyChart = new Chart(ctx, {
      type: "line",
      data: chartData,
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#9aa4b2", maxTicksLimit: 15 } },
          y: { ticks: { color: "#9aa4b2", stepSize: 1 }, min: 0, max: 1 },
        },
      },
    });
  }
}

async function init() {
  try {
    await loadStats();
    await loadLatestPredictions();

    if (latestPredictions.length === 0) {
      showError("No predictions logged yet. Run src/db/score_latest_month.py first.");
      return;
    }

    const slider = document.getElementById("top-n-slider");
    const sliderValue = document.getElementById("top-n-value");
    renderRiskTableAndChart(parseInt(slider.value, 10));
    slider.addEventListener("input", () => {
      sliderValue.textContent = slider.value;
      renderRiskTableAndChart(parseInt(slider.value, 10));
    });

    populateChemicalSelect();
    const select = document.getElementById("chemical-select");
    await loadChemicalHistory(select.value);
    select.addEventListener("change", () => loadChemicalHistory(select.value));
  } catch (e) {
    showError(`Failed to load data from the API: ${e.message}`);
  }
}

init();