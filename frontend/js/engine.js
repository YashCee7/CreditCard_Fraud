/**
 * engine.js — FraudGuard Prediction Engine
 *
 * This file simulates the XGBoost model's behaviour using the
 * same feature weights learned from the Kaggle credit-card dataset.
 *
 * HOW TO CONNECT A REAL BACKEND
 * ─────────────────────────────
 * When you deploy a FastAPI/Flask backend that serves your saved
 * XGBoost model (.pkl), replace the body of `scoreSingleRow()`
 * with a fetch() call:
 *
 *   const res  = await fetch('http://localhost:8000/predict', {
 *     method:  'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body:    JSON.stringify({ features: row })
 *   });
 *   const data = await res.json();
 *   return data.probability;  // float 0–1
 *
 * Everything else in the frontend stays exactly the same.
 */

'use strict';

// ── CSV PARSER ──────────────────────────────────────────────────────────────
/**
 * Parse a CSV string into { headers: string[], rows: object[] }
 * Handles quoted fields and trims whitespace.
 */
function parseCSV(text) {
  const lines   = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const rows    = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    // Simple split — handles most CSV exports from pandas/Excel
    const vals = line.split(',');
    const row  = {};
    headers.forEach((h, j) => {
      const raw = (vals[j] ?? '').trim().replace(/^"|"$/g, '');
      const num = parseFloat(raw);
      row[h]    = isNaN(num) ? raw : num;
    });
    rows.push(row);
  }
  return { headers, rows };
}

// ── FEATURE FILLING ─────────────────────────────────────────────────────────
/**
 * Ensure a row has all 30 features (Time, Amount, V1–V28).
 * Missing V-features are filled with small Gaussian-like noise
 * (simulates what an unseen row would look like in practice).
 */
function fillMissingFeatures(row) {
  const filled = { ...row };
  for (let n = 1; n <= 28; n++) {
    const key = 'V' + n;
    if (typeof filled[key] !== 'number' || isNaN(filled[key])) {
      // Small random value around 0 (similar to PCA-transformed features)
      filled[key] = (Math.random() - 0.5) * 0.6;
    }
  }
  if (typeof filled['Time']   !== 'number') filled['Time']   = 0;
  if (typeof filled['Amount'] !== 'number') filled['Amount'] = 0;
  return filled;
}

// ── FRAUD SCORING ENGINE ────────────────────────────────────────────────────
/**
 * Approximate XGBoost fraud probability using feature weights
 * derived from the Kaggle creditcard.csv dataset.
 *
 * Feature importance order (from RF feature_importances_):
 * V14 > V10 > V12 > V17 > V4 > V7 > V11 > V3 > V16 > V2
 *
 * These weights replicate the direction and relative magnitude
 * of XGBoost's learned splits on the same dataset.
 *
 * @param {object} row — object with Time, Amount, V1–V28 keys
 * @returns {number} probability in [0, 1]
 */
async function scoreSingleRow(row) {
  const res = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      features: Object.values(row)
    })
  });

  const data = await res.json();
  return data.probability;
}

// ── BATCH SCORING ───────────────────────────────────────────────────────────
/**
 * Score an array of raw CSV rows and return enriched result objects.
 *
 * @param {object[]} rawRows   — parsed CSV rows
 * @param {number}   threshold — decision threshold (default 0.25)
 * @returns {object[]} result rows with id, amount, time, prob, isFraud, action
 */
async function scoreBatch(rawRows, threshold = 0.25) {
  const results = [];

  for (let i = 0; i < rawRows.length; i++) {
    const row = rawRows[i];
    const filled = fillMissingFeatures(row);

    const prob = await scoreSingleRow(filled);
    const isFraud = prob > threshold;

    results.push({
      id: row['Transaction_ID'] || ('TXN_' + String(i + 1).padStart(5, '0')),
      amount: typeof row['Amount'] === 'number' ? row['Amount'] : Math.random() * 300 + 5,
      time: typeof row['Time'] === 'number' ? row['Time'] : Math.random() * 172800,
      prob: parseFloat(prob.toFixed(4)),
      isFraud,
      action: 'pending',
      rowIndex: i,
    });
  }

  return results;
}

// ── DEMO DATA GENERATOR ─────────────────────────────────────────────────────
/**
 * Generate n synthetic transactions that follow the real dataset's
 * distribution (~0.17% fraud rate, bimodal amount distribution).
 * Used when no CSV is uploaded so the demo can run immediately.
 */
function generateDemoData(n = 500) {
  const rows = [];

  for (let i = 0; i < n; i++) {
    // ~0.5% fraud seed (slightly elevated for demo visibility)
    const isFraudSeed = Math.random() < 0.005;

    const row = {
      Time:   Math.random() * 172800,
      Amount: isFraudSeed
        ? (Math.random() < 0.5 ? Math.random() * 50 + 1 : Math.random() * 5000 + 1000)
        : Math.random() * 400 + 5,
    };

    for (let v = 1; v <= 28; v++) {
      const base = (Math.random() - 0.5) * 2;
      // Inject fraud-like signal into key V-features for seeded fraud rows
      const fraudSignal = isFraudSeed ? (
        v === 14 ? -4.5 + Math.random() :
        v === 10 ? -3.8 + Math.random() :
        v === 12 ? -3.2 + Math.random() :
        v === 4  ?  3.0 + Math.random() :
        v === 11 ?  2.5 + Math.random() : base
      ) : base;

      row['V' + v] = fraudSignal;
    }

    rows.push(row);
  }

  return rows;
}

// ── FILE VALIDATION ─────────────────────────────────────────────────────────
/**
 * Validate a parsed CSV for required columns.
 *
 * @param {object[]} rows
 * @returns {object} { valid: bool, checks: [...], warnings: [...], errors: [...] }
 */
function validateCSV(rows) {
  if (!rows || rows.length === 0) {
    return { valid: false, errors: ['File is empty — no rows found.'] };
  }

  const keys    = Object.keys(rows[0]);
  const checks  = [];
  const warnings= [];
  const errors  = [];

  // Format
  checks.push({ ok: true, msg: 'File format: CSV ✓' });

  // Row count
  checks.push({ ok: true, msg: `Rows detected: ${rows.length.toLocaleString()}` });

  // Time
  const hasTime = keys.includes('Time');
  if (hasTime) checks.push({ ok: true,  msg: 'Time column: Present ✓' });
  else          errors.push('Time column is MISSING — required');

  // Amount
  const hasAmount = keys.includes('Amount');
  if (hasAmount) checks.push({ ok: true,  msg: 'Amount column: Present ✓' });
  else            errors.push('Amount column is MISSING — required');

  // V-features
  const presentV = Array.from({ length: 28 }, (_, i) => 'V' + (i + 1))
                        .filter(v => keys.includes(v));
  const missingV = Array.from({ length: 28 }, (_, i) => 'V' + (i + 1))
                        .filter(v => !keys.includes(v));
  const vMsg = `V-features: ${presentV.length}/28 present`;
  if (presentV.length === 28) checks.push({ ok: true, msg: vMsg });
  else if (presentV.length >= 20) warnings.push(vMsg + ` (missing: ${missingV.slice(0,6).join(', ')}${missingV.length>6?'…':''})`);
  else errors.push(vMsg + ` — too many missing`);

  // Class column warning
  if (keys.includes('Class')) {
    warnings.push('Class column detected — will be ignored (we predict this)');
  }

  // Transaction ID warning
  if (!keys.includes('Transaction_ID')) {
    warnings.push('No Transaction_ID column — rows will be numbered automatically');
  }

  const estSeconds = Math.max(1, Math.round(rows.length / 8000));
  checks.push({ ok: true, msg: `Estimated processing time: ~${estSeconds}s` });

  return {
    valid:    errors.length === 0,
    checks,
    warnings,
    errors,
  };
}
