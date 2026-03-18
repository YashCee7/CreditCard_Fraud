/**
 * app.js — FraudGuard Application Controller
 *
 * Wires together engine.js, ui.js, and charts.js.
 * Handles all user interactions, state, and rendering.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════
const State = {
  // Single check
  singleFileData: null,

  // Bulk analysis
  bulkRawData:  [],
  bulkResults:  [],
  bulkThreshold: 0.25,

  // Table pagination & filtering
  currentPage: 1,
  PAGE_SIZE:   10,
  filterMode:  'all',     // 'all' | 'fraud' | 'high' | 'legit'
  searchQuery: '',
  sortCol:     'prob',
  sortDir:     -1,        // -1 = descending, 1 = ascending
};

// ═══════════════════════════════════════════════════════════════
// INIT — runs after all scripts are loaded
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initSliders();
  initDropZones();
  initButtons();
});

// ── Sliders ──────────────────────────────────────────────────
function initSliders() {
  const singleSlider = document.getElementById('single-threshold');
  const bulkSlider   = document.getElementById('bulk-threshold');

  if (singleSlider) {
    singleSlider.addEventListener('input', () => {
      syncSlider(singleSlider, ['threshold-display', 'threshold-display2']);
    });
    syncSlider(singleSlider, ['threshold-display', 'threshold-display2']);
  }

  if (bulkSlider) {
    bulkSlider.addEventListener('input', () => {
      syncSlider(bulkSlider, ['bulk-thresh-val']);
    });
    syncSlider(bulkSlider, ['bulk-thresh-val']);
  }
}

// ── Drop Zones ───────────────────────────────────────────────
function initDropZones() {
  // Single file drop
  setupDropZone('single-dropzone', file => processSingleCSV(file));

  // Single file input change (click to browse)
  const singleInput = document.getElementById('single-file-input');
  if (singleInput) {
    singleInput.addEventListener('change', e => {
      if (e.target.files[0]) processSingleCSV(e.target.files[0]);
    });
  }

  // Bulk file drop
  setupDropZone('bulk-dropzone', file => processBulkCSV(file));

  // Bulk file input change
  const bulkInput = document.getElementById('bulk-file-input');
  if (bulkInput) {
    bulkInput.addEventListener('change', e => {
      if (e.target.files[0]) processBulkCSV(e.target.files[0]);
    });
  }
}

// ── Buttons ──────────────────────────────────────────────────
function initButtons() {
  // Advanced toggle
  const advToggle = document.getElementById('adv-toggle');
  if (advToggle) {
    advToggle.addEventListener('click', () => {
      const content = document.getElementById('advanced-content');
      const arrow   = document.getElementById('adv-arrow');
      const isOpen  = content.classList.toggle('open');
      arrow.textContent = isOpen ? '▼' : '▶';
    });
  }

  // Single check
  const singleBtn = document.getElementById('single-check-btn');
  if (singleBtn) singleBtn.addEventListener('click', runSingleCheck);

  // Bulk validate
  const validateBtn = document.getElementById('validate-btn');
  if (validateBtn) validateBtn.addEventListener('click', validateBulkFile);

  // Bulk run
  const runBtn = document.getElementById('run-analysis-btn');
  if (runBtn) runBtn.addEventListener('click', runBulkAnalysis);

  // New analysis (reset bulk)
  const newBtn = document.getElementById('new-analysis-btn');
  if (newBtn) newBtn.addEventListener('click', resetBulk);
}

// ═══════════════════════════════════════════════════════════════
// SINGLE CHECK
// ═══════════════════════════════════════════════════════════════

function processSingleCSV(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showToast('⚠️ Please upload a CSV file'); return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const { rows } = parseCSV(e.target.result);
      State.singleFileData = rows[0] || null;

      // Auto-fill Amount & Time from file if not yet entered
      const amtInput  = document.getElementById('input-amount');
      const timeInput = document.getElementById('input-time');
      if (State.singleFileData?.Amount && !amtInput.value)
        amtInput.value = State.singleFileData.Amount;
      if (State.singleFileData?.Time && !timeInput.value)
        timeInput.value = State.singleFileData.Time;

      // Visual feedback
      const zone = document.getElementById('single-dropzone');
      zone.classList.add('file-loaded');
      document.getElementById('single-drop-icon').textContent  = '✅';
      document.getElementById('single-drop-title').textContent = '✓ ' + file.name + ' loaded';
      showToast('✓ File loaded — V1–V28 ready');
    } catch (err) {
      showToast('❌ Error reading file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

async function runSingleCheck() {
  const amount = parseFloat(document.getElementById('input-amount').value);
  const time   = parseFloat(document.getElementById('input-time').value);
  const threshold = parseFloat(document.getElementById('single-threshold').value);

  if (isNaN(amount) || isNaN(time)) {
    showToast('⚠️ Please enter both Amount and Time'); return;
  }

  showLoading('Analyzing transaction…', [
    'Scaling features (StandardScaler)',
    'Running XGBoost inference',
    'Applying threshold (' + threshold + ')',
  ]);

  // Stagger loading step indicators
  const steps = [300, 700, 1100];
  steps.forEach((delay, i) => setTimeout(() => setLoadingStep(i), delay));

  setTimeout(async () => {
  hideLoading();

  const row = { ...(State.singleFileData || {}), Amount: amount, Time: time };
  const filled = fillMissingFeatures(row);

  const prob = await scoreSingleRow(filled);
  const isFraud = prob > threshold;

  renderSingleResult(prob, isFraud, threshold, amount);
  }, 1450);
}

function renderSingleResult(prob, isFraud, threshold, amount) {
  const pct      = (prob * 100).toFixed(1);
  const risk     = getRiskInfo(prob, threshold);
  const threshPct = (threshold * 100).toFixed(0);
  const cardClass = isFraud ? 'fraud' : 'legit';
  const verdictIcon = isFraud ? '🔴' : '🟢';
  const verdictText = isFraud ? 'FRAUDULENT TRANSACTION' : 'LEGITIMATE TRANSACTION';

  const explainText = isFraud
    ? `This transaction has a <strong style="color:var(--red)">${pct}%</strong> probability of being fraudulent. Amount ₹${amount.toFixed(2)} is flagged at <strong>${risk.label.replace(/🔴|🟡|🟠|🟢/g, '').trim()}</strong> risk. Immediate action is recommended.`
    : `This transaction has a <strong style="color:var(--green)">${(100 - parseFloat(pct)).toFixed(1)}%</strong> probability of being legitimate. Fraud probability (${pct}%) is below threshold (${threshPct}%). No action required.`;

  const html = `
    <div class="result-card ${cardClass}">
      <div class="result-verdict ${cardClass}">
        <span>${verdictIcon}</span>
        ${verdictText}
      </div>

      <div class="result-meta">
        Confidence: <strong style="color:${risk.color}">${pct}%</strong>
        &nbsp;·&nbsp;
        Risk Level: <strong style="color:${risk.color}">${risk.label.replace(/🔴|🟡|🟠|🟢/g, '').trim()}</strong>
        &nbsp;·&nbsp;
        Threshold: <strong style="font-family:var(--font-mono)">${threshold}</strong>
      </div>

      <!-- Probability Gauge -->
      <div class="gauge-wrap">
        <div class="gauge-label">FRAUD PROBABILITY GAUGE</div>
        <div class="gauge-bar">
          <div class="gauge-fill ${cardClass}" id="gauge-fill" style="width: 0%;"></div>
          <div class="gauge-threshold" style="left: ${threshPct}%;">
            <div class="gauge-threshold-label">threshold&nbsp;${threshold}</div>
          </div>
        </div>
        <div class="gauge-ticks">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
      </div>

      <!-- Plain English Explanation -->
      <div class="result-explain ${cardClass}">${explainText}</div>

      <!-- Action Buttons -->
      <div class="action-row">
        <button class="action-btn block"       onclick="logSingleAction(this, 'Transaction Blocked')">🚫 Block Transaction</button>
        <button class="action-btn investigate" onclick="logSingleAction(this, 'Sent for Investigation')">🔎 Investigate Further</button>
        <button class="action-btn override"    onclick="logSingleAction(this, 'Marked as Legitimate')">✅ Override as Legit</button>
      </div>
    </div>`;

  const outputSection = document.getElementById('single-output');
  document.getElementById('result-card').innerHTML = html;
  outputSection.classList.remove('hidden');
  outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Animate gauge fill after paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = document.getElementById('gauge-fill');
      if (fill) fill.style.width = (prob * 100) + '%';
    }, 80);
  });
}

function logSingleAction(btn, actionLabel) {
  if (btn.classList.contains('logged')) return;
  document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('logged'));
  btn.classList.add('logged');
  showToast('✓ Action logged: ' + actionLabel);
}

// ═══════════════════════════════════════════════════════════════
// BULK ANALYSIS
// ═══════════════════════════════════════════════════════════════

function processBulkCSV(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showToast('⚠️ Please upload a CSV file'); return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const { rows } = parseCSV(e.target.result);
      State.bulkRawData = rows;

      const zone = document.getElementById('bulk-dropzone');
      zone.classList.add('file-loaded');
      document.getElementById('bulk-drop-icon').textContent  = '✅';
      document.getElementById('bulk-drop-title').textContent =
        '✓ ' + file.name + ' — ' + rows.length.toLocaleString() + ' rows';

      showToast(`✓ Loaded ${rows.length.toLocaleString()} transactions`);
    } catch (err) {
      showToast('❌ Error reading file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function validateBulkFile() {
  if (!State.bulkRawData.length) {
    State.bulkRawData = generateDemoData(500);
    showToast('No file uploaded — using 500 demo transactions');
  }

  const result = validateCSV(State.bulkRawData);
  const box    = document.getElementById('validation-box');

  const checksHTML = result.checks
    .map(c => `<div class="val-item"><span class="val-icon val-ok">✅</span>${c.msg}</div>`)
    .join('');
  const warnHTML = result.warnings
    .map(w => `<div class="val-item"><span class="val-icon val-warn">⚠️</span>${w}</div>`)
    .join('');
  const errHTML = result.errors
    .map(e => `<div class="val-item"><span class="val-icon val-err">❌</span>${e}</div>`)
    .join('');

  const proceedBtn = result.valid
    ? `<button class="btn btn-primary" onclick="runBulkAnalysis()">🚀 Proceed with Analysis</button>`
    : '';

  box.innerHTML = `
    <div class="validation-box">
      <div class="validation-title">📋 File Validation Report</div>
      ${checksHTML}${warnHTML}${errHTML}
      <div class="val-actions">
        <button class="btn btn-secondary" onclick="resetDropZone()">◀ Re-upload</button>
        ${proceedBtn}
      </div>
    </div>`;
  box.classList.remove('hidden');
}

function resetDropZone() {
  State.bulkRawData = [];
  const zone = document.getElementById('bulk-dropzone');
  zone.classList.remove('file-loaded');
  document.getElementById('bulk-drop-icon').textContent  = '📁';
  document.getElementById('bulk-drop-title').textContent = 'Drop CSV file here or click to browse';
  document.getElementById('validation-box').classList.add('hidden');
}

async function runBulkAnalysis() {
  if (!State.bulkRawData.length) {
    State.bulkRawData = generateDemoData(500);
    showToast('No file uploaded — using 500 demo transactions');
  }

  State.bulkThreshold = parseFloat(document.getElementById('bulk-threshold').value);
  State.currentPage   = 1;
  State.filterMode    = 'all';
  State.searchQuery   = '';
  State.sortCol       = 'prob';
  State.sortDir       = -1;

  const n = State.bulkRawData.length;
  showLoading(`Analyzing ${n.toLocaleString()} transactions…`, [
    'Scaling Time & Amount (StandardScaler)',
    'Running XGBoost inference',
    `Applying threshold (${State.bulkThreshold})`,
    'Computing statistics',
    'Generating charts',
  ]);

  let step = 0;
  const stepInterval = setInterval(() => {
    setLoadingStep(step++);
    if (step >= 5) clearInterval(stepInterval);
  }, 280);

  setTimeout(async () => {
  hideLoading();

  State.bulkResults = await scoreBatch(State.bulkRawData, State.bulkThreshold);

  document.getElementById('bulk-upload-step').classList.add('hidden');
  document.getElementById('bulk-results-step').classList.remove('hidden');

  renderBulkResults();
  }, 1650);
}

function resetBulk() {
  State.bulkRawData  = [];
  State.bulkResults  = [];

  document.getElementById('bulk-results-step').classList.add('hidden');
  document.getElementById('bulk-upload-step').classList.remove('hidden');

  // Reset drop zone
  const zone = document.getElementById('bulk-dropzone');
  zone.classList.remove('file-loaded');
  document.getElementById('bulk-drop-icon').textContent  = '📁';
  document.getElementById('bulk-drop-title').textContent = 'Drop CSV file here or click to browse';
  document.getElementById('validation-box').classList.add('hidden');

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ═══════════════════════════════════════════════════════════════
// BULK RESULTS RENDERING
// ═══════════════════════════════════════════════════════════════

function renderBulkResults() {
  const results    = State.bulkResults;
  const fraudRows  = results.filter(r => r.isFraud);
  const totalAmt   = fraudRows.reduce((s, r) => s + r.amount, 0);
  const topRisk    = [...fraudRows].sort((a, b) => b.prob - a.prob)[0];

  // ── Summary Cards ──
  document.getElementById('bulk-summary').innerHTML = `
    <div class="summary-grid">
      <div class="summary-card">
        <div class="summary-label">Total Transactions</div>
        <div class="summary-val accent">${results.length.toLocaleString()}</div>
        <div class="summary-sub">processed</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">Flagged as Fraud</div>
        <div class="summary-val red">${fraudRows.length.toLocaleString()}</div>
        <div class="summary-sub">transactions</div>
      </div>
      <div class="summary-card">
        <div class="summary-label">Fraud Rate</div>
        <div class="summary-val">${(fraudRows.length / results.length * 100).toFixed(2)}%</div>
        <div class="summary-sub">of all transactions</div>
      </div>
      <div class="summary-card highlight">
        <div class="summary-label">Amount at Risk</div>
        <div class="summary-val accent">${formatAmount(totalAmt)}</div>
        <div class="summary-sub">${formatAmountFull(totalAmt)}</div>
      </div>
    </div>
    ${topRisk ? `
    <div class="top-risk-bar">
      🔴 <strong>HIGHEST RISK:</strong>
      ${topRisk.id}
      &nbsp;·&nbsp; ${formatAmountFull(topRisk.amount)}
      &nbsp;·&nbsp; <strong style="color:var(--red)">${(topRisk.prob * 100).toFixed(1)}% confidence</strong>
      &nbsp;·&nbsp; <span style="color:var(--text3)">Needs immediate action</span>
    </div>` : ''}
    <hr class="divider" />`;

  // ── Charts ──
  renderAllCharts('bulk-charts', results, State.bulkThreshold);

  // ── Table ──
  renderTable();

  // ── Downloads ──
  document.getElementById('bulk-downloads').innerHTML = `
    <div class="download-section">
      <div class="download-title">📥 Export Results</div>
      <div class="download-grid">
        <div class="download-card" onclick="downloadFlagged()">
          <div class="download-card-icon">📄</div>
          <div class="download-card-title">Flagged Transactions</div>
          <div class="download-card-sub">CSV · ${fraudRows.length} rows · For case management</div>
        </div>
        <div class="download-card" onclick="downloadFull()">
          <div class="download-card-icon">📊</div>
          <div class="download-card-title">Full Results</div>
          <div class="download-card-sub">CSV · All ${results.length.toLocaleString()} rows + probability column</div>
        </div>
        <div class="download-card" onclick="openPDFReport()">
          <div class="download-card-icon">📑</div>
          <div class="download-card-title">PDF Summary Report</div>
          <div class="download-card-sub">For management presentation</div>
        </div>
        <div class="download-card" onclick="showToast('Email requires a backend integration')">
          <div class="download-card-icon">📧</div>
          <div class="download-card-title">Email Report</div>
          <div class="download-card-sub">Send to team (requires backend)</div>
        </div>
      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════
// TABLE
// ═══════════════════════════════════════════════════════════════

function renderTable() {
  const threshold = State.bulkThreshold;

  // ── Filter ──
  let filtered = State.bulkResults;
  if      (State.filterMode === 'fraud') filtered = filtered.filter(r => r.isFraud);
  else if (State.filterMode === 'legit') filtered = filtered.filter(r => !r.isFraud);
  else if (State.filterMode === 'high')  filtered = filtered.filter(r => r.prob > 0.80);

  if (State.searchQuery) {
    const q = State.searchQuery.toLowerCase();
    filtered = filtered.filter(r => r.id.toLowerCase().includes(q));
  }

  // ── Sort ──
  filtered = [...filtered].sort((a, b) => {
    const dir = State.sortDir;
    if (State.sortCol === 'prob')   return dir * (b.prob - a.prob);
    if (State.sortCol === 'amount') return dir * (b.amount - a.amount);
    if (State.sortCol === 'time')   return dir * (b.time - a.time);
    return 0;
  });

  // ── Paginate ──
  const total  = filtered.length;
  const pages  = Math.max(1, Math.ceil(total / State.PAGE_SIZE));
  State.currentPage = Math.min(State.currentPage, pages);
  const start  = (State.currentPage - 1) * State.PAGE_SIZE;
  const pageRows = filtered.slice(start, start + State.PAGE_SIZE);

  const fraudCount = State.bulkResults.filter(r => r.isFraud).length;
  const allCount   = State.bulkResults.length;
  const highCount  = State.bulkResults.filter(r => r.prob > 0.80).length;
  const legitCount = State.bulkResults.filter(r => !r.isFraud).length;

  // Sort arrow helper
  const arrow = col => State.sortCol === col ? (State.sortDir === -1 ? ' ↓' : ' ↑') : ' ↕';

  // ── Row HTML ──
  const rowsHTML = pageRows.length === 0
    ? `<tr><td colspan="6" style="text-align:center;color:var(--text3);padding:32px;">
         No transactions match the current filter
       </td></tr>`
    : pageRows.map(r => {
        const risk   = getRiskInfo(r.prob, threshold);
        const probPct = (r.prob * 100).toFixed(1);
        const timeStr = formatTime(r.time);
        return `
          <tr>
            <td style="font-family:var(--font-mono);font-size:12px;color:var(--text2)">${r.id}</td>
            <td><strong>${formatAmountFull(r.amount)}</strong></td>
            <td style="font-family:var(--font-mono);font-size:12px;color:var(--text2)">${timeStr}</td>
            <td>
              <div class="prob-bar-wrap">
                <div class="prob-bar">
                  <div class="prob-fill" style="width:${probPct}%;background:${risk.color}"></div>
                </div>
                <span class="prob-text" style="color:${risk.color}">${probPct}%</span>
              </div>
            </td>
            <td><span class="risk-badge ${risk.cssClass}">${risk.label}</span></td>
            <td>
              <select class="action-select"
                onchange="updateAction('${r.id}', this.value)">
                <option value="pending"    ${r.action==='pending'    ?'selected':''}>⏳ Pending</option>
                <option value="block"      ${r.action==='block'      ?'selected':''}>🚫 Block</option>
                <option value="investigate"${r.action==='investigate'?'selected':''}>🔎 Investigate</option>
                <option value="legit"      ${r.action==='legit'      ?'selected':''}>✅ Mark Legit</option>
                <option value="escalate"   ${r.action==='escalate'   ?'selected':''}>📤 Escalate</option>
              </select>
            </td>
          </tr>`;
      }).join('');

  // ── Page buttons ──
  const maxPageBtns = 5;
  const startPage = Math.max(1, Math.min(pages - maxPageBtns + 1, State.currentPage - 2));
  const endPage   = Math.min(pages, startPage + maxPageBtns - 1);
  let pageBtnsHTML = `
    <button class="page-btn" onclick="changePage(-1)"
      ${State.currentPage <= 1 ? 'disabled' : ''}>‹</button>`;
  for (let p = startPage; p <= endPage; p++) {
    pageBtnsHTML += `
      <button class="page-btn${p === State.currentPage ? ' active' : ''}"
        onclick="goToPage(${p})">${p}</button>`;
  }
  pageBtnsHTML += `
    <button class="page-btn" onclick="changePage(1)"
      ${State.currentPage >= pages ? 'disabled' : ''}>›</button>`;

  document.getElementById('bulk-table-section').innerHTML = `
    <div class="table-header">
      <div style="font-weight:700;font-size:15px;">
        🚨 Transaction Results
        <span style="color:var(--red);font-family:var(--font-mono);font-size:13px;">
          (${fraudCount} fraud flagged)
        </span>
      </div>
      <div class="table-controls">
        <input class="search-input" placeholder="Search TXN ID…"
          value="${State.searchQuery}"
          oninput="tableSearch(this.value)" />
        <select class="filter-select" onchange="tableFilter(this.value)">
          <option value="all"   ${State.filterMode==='all'   ?'selected':''}>All (${allCount})</option>
          <option value="fraud" ${State.filterMode==='fraud' ?'selected':''}>Fraud Only (${fraudCount})</option>
          <option value="high"  ${State.filterMode==='high'  ?'selected':''}>High Risk (${highCount})</option>
          <option value="legit" ${State.filterMode==='legit' ?'selected':''}>Legit Only (${legitCount})</option>
        </select>
        <button class="btn btn-secondary"
          style="padding:8px 14px;font-size:12px;"
          onclick="downloadFlagged()">↓ Flagged CSV</button>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th onclick="tableSort('id')">TXN ID</th>
            <th onclick="tableSort('amount')">AMOUNT${arrow('amount')}</th>
            <th onclick="tableSort('time')">TIME${arrow('time')}</th>
            <th onclick="tableSort('prob')">PROBABILITY${arrow('prob')}</th>
            <th>RISK</th>
            <th>ACTION</th>
          </tr>
        </thead>
        <tbody>${rowsHTML}</tbody>
      </table>
    </div>

    <div class="pagination">
      <div class="page-info">
        Showing ${total === 0 ? 0 : start + 1}–${Math.min(start + State.PAGE_SIZE, total)}
        of ${total.toLocaleString()}
      </div>
      <div class="page-btns">${pageBtnsHTML}</div>
    </div>`;
}

// ── Table interaction handlers ──
function tableSort(col) {
  if (State.sortCol === col) State.sortDir *= -1;
  else { State.sortCol = col; State.sortDir = -1; }
  State.currentPage = 1;
  renderTable();
}

function tableFilter(val) {
  State.filterMode  = val;
  State.currentPage = 1;
  renderTable();
}

function tableSearch(val) {
  State.searchQuery = val;
  State.currentPage = 1;
  renderTable();
}

function changePage(delta) {
  State.currentPage += delta;
  renderTable();
}

function goToPage(p) {
  State.currentPage = p;
  renderTable();
}

function updateAction(id, val) {
  const row = State.bulkResults.find(r => r.id === id);
  if (row) {
    row.action = val;
    showToast('✓ Action updated: ' + id);
  }
}

// ═══════════════════════════════════════════════════════════════
// DOWNLOADS
// ═══════════════════════════════════════════════════════════════

function downloadFlagged() {
  const fraud = State.bulkResults.filter(r => r.isFraud);
  if (!fraud.length) { showToast('No fraud transactions to export'); return; }

  const rows = fraud.map(r => ({
    transaction_id:    r.id,
    amount:            r.amount.toFixed(2),
    time_seconds:      r.time.toFixed(0),
    time_of_day:       formatTime(r.time),
    fraud_probability: (r.prob * 100).toFixed(2) + '%',
    risk_level:        r.prob > 0.8 ? 'HIGH' : r.prob > 0.5 ? 'MEDIUM' : 'LOW',
    action:            r.action,
  }));

  const csv = rowsToCSV(rows, Object.keys(rows[0]));
  downloadCSVFile(csv, 'fraudguard_flagged_' + Date.now() + '.csv');
  showToast(`✓ Downloaded ${fraud.length} flagged transactions`);
}

function downloadFull() {
  const rows = State.bulkResults.map(r => ({
    transaction_id:    r.id,
    amount:            r.amount.toFixed(2),
    time_seconds:      r.time.toFixed(0),
    fraud_probability: (r.prob * 100).toFixed(2) + '%',
    prediction:        r.isFraud ? 'FRAUD' : 'LEGIT',
    action:            r.action,
  }));

  const csv = rowsToCSV(rows, Object.keys(rows[0]));
  downloadCSVFile(csv, 'fraudguard_full_' + Date.now() + '.csv');
  showToast(`✓ Downloaded all ${State.bulkResults.length.toLocaleString()} results`);
}

function openPDFReport() {
  const results   = State.bulkResults;
  const fraud     = results.filter(r => r.isFraud);
  const totalAmt  = fraud.reduce((s, r) => s + r.amount, 0);
  const topRisk   = [...fraud].sort((a, b) => b.prob - a.prob)[0];
  const now       = new Date().toLocaleString();
  const threshold = State.bulkThreshold;

  const topRowsHTML = fraud.slice(0, 20).map(r => `
    <tr>
      <td style="font-family:monospace">${r.id}</td>
      <td>${formatAmountFull(r.amount)}</td>
      <td>${formatTime(r.time)}</td>
      <td>${(r.prob * 100).toFixed(1)}%</td>
      <td><span style="background:#ffebee;color:#c62828;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">FRAUD</span></td>
    </tr>`).join('');

  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html>
<html><head><title>FraudGuard Report</title>
<style>
  body{font-family:Arial,sans-serif;padding:40px;color:#111;max-width:720px;margin:0 auto}
  h1{font-size:24px;margin-bottom:4px}
  .sub{color:#666;font-size:13px;margin-bottom:32px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
  .card{border:1px solid #ddd;border-radius:8px;padding:18px}
  .val{font-size:26px;font-weight:800}
  .lbl{font-size:11px;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px}
  .red{color:#e53935} .blue{color:#1565c0}
  .alert{border:1px solid #ffcdd2;background:#fff8f8;border-radius:8px;padding:16px;margin-bottom:24px}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:16px}
  th{background:#f5f5f5;padding:8px 12px;text-align:left;font-size:11px;color:#888;text-transform:uppercase;border-bottom:1px solid #ddd}
  td{padding:8px 12px;border-bottom:1px solid #eee}
  .footer{margin-top:40px;color:#aaa;font-size:11px;border-top:1px solid #eee;padding-top:20px}
</style></head><body>
<h1>🔒 FraudGuard — Analysis Report</h1>
<div class="sub">Generated: ${now} &nbsp;·&nbsp; Model: XGBoost &nbsp;·&nbsp; Threshold: ${threshold}</div>
<div class="grid">
  <div class="card"><div class="lbl">Total Transactions</div><div class="val blue">${results.length.toLocaleString()}</div></div>
  <div class="card"><div class="lbl">Fraud Detected</div><div class="val red">${fraud.length.toLocaleString()}</div></div>
  <div class="card"><div class="lbl">Fraud Rate</div><div class="val">${(fraud.length/results.length*100).toFixed(2)}%</div></div>
  <div class="card"><div class="lbl">Amount at Risk</div><div class="val red">${formatAmountFull(totalAmt)}</div></div>
</div>
${topRisk ? `<div class="alert">
  <div class="lbl">⚠️ Highest Risk Transaction</div>
  <strong>${topRisk.id}</strong> &nbsp;·&nbsp; ${formatAmountFull(topRisk.amount)}
  &nbsp;·&nbsp; <strong>${(topRisk.prob*100).toFixed(1)}% confidence</strong>
</div>` : ''}
<h3>Top Flagged Transactions (up to 20)</h3>
<table>
<tr><th>TXN ID</th><th>Amount</th><th>Time</th><th>Probability</th><th>Status</th></tr>
${topRowsHTML}
</table>
<div class="footer">
  This report was generated by FraudGuard using a simulated XGBoost fraud detection engine.
  All predictions are probabilistic and should be reviewed by a qualified analyst before taking action.
  Model architecture based on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions).
</div>
</body></html>`);
  win.document.close();
  setTimeout(() => win.print(), 400);
  showToast('✓ PDF report opened for printing / Save as PDF');
}
