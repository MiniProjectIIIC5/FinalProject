/* ══════════════════════════════════════════════
   PROFILE VERIFIER — FRONTEND SCRIPT
══════════════════════════════════════════════ */

const API = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : 'https://finalproject-14we.onrender.com/api';
const cors = require("cors");
app.use(cors({
  origin: "https://profileverifier.netlify.app",
  methods: ["GET", "POST"],
  credentials: true
}));
// ── State ──────────────────────────────────────
let token       = localStorage.getItem('pv_token')  || null;
let currentUser = JSON.parse(localStorage.getItem('pv_user') || 'null');
let historyData = [];
let currentVerifyUrl = null;
let selectedPlatform = null;

// ── Boot ───────────────────────────────────────
(function boot() {
  if (token && currentUser) {
    showApp();
  } else {
    showAuth();
  }
})();

function showAuth() {
  document.getElementById('auth-container').style.display = 'flex';
  document.getElementById('app-container').style.display  = 'none';
}

function showApp() {
  document.getElementById('auth-container').style.display = 'none';
  document.getElementById('app-container').style.display  = 'block';
  // Set user display
  const name = currentUser?.username || 'User';
  document.getElementById('userNameDisplay').textContent = name;
  document.getElementById('userAvatar').textContent = name.charAt(0).toUpperCase();
  // Attach nav listeners
  attachNav();
  loadStats();
  loadHistory();
}

/* ══════════════════════════════════════════════
   AUTH
══════════════════════════════════════════════ */
function showSignup() {
  document.getElementById('signinForm').classList.remove('active');
  document.getElementById('signupForm').classList.add('active');
  clearFieldErrors();
}
function showSignin() {
  document.getElementById('signupForm').classList.remove('active');
  document.getElementById('signinForm').classList.add('active');
  clearFieldErrors();
}

function clearFieldErrors() {
  document.querySelectorAll('.field-err').forEach(el => el.textContent = '');
}
function showFieldErr(id, msg) {
  const el = document.getElementById('err-' + id);
  if (el) el.textContent = msg;
}

function setBtnLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  btn.querySelector('.btn-text').style.display    = loading ? 'none'  : 'inline';
  btn.querySelector('.btn-spinner').style.display = loading ? 'inline': 'none';
}

// Sign Up
document.getElementById('signupForm').addEventListener('submit', async e => {
  e.preventDefault();
  clearFieldErrors();
  const username = document.getElementById('signupUsername').value.trim();
  const email    = document.getElementById('signupEmail').value.trim();
  const password = document.getElementById('signupPassword').value;

  let valid = true;
  if (!username) { showFieldErr('signupUsername', 'Username is required'); valid=false; }
  if (!email)    { showFieldErr('signupEmail',    'Email is required');    valid=false; }
  if (password.length < 6) { showFieldErr('signupPassword', 'Min 6 characters'); valid=false; }
  if (!valid) return;

  setBtnLoading('signupBtn', true);
  try {
    const res  = await fetch(`${API}/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();
    if (res.ok) {
      showToast('Account created! Please sign in.', 'success');
      e.target.reset();
      showSignin();
    } else {
      showToast(data.message || 'Sign up failed', 'error');
    }
  } catch {
    showToast('Network error. Is the server running?', 'error');
  }
  setBtnLoading('signupBtn', false);
});

// Sign In
document.getElementById('signinForm').addEventListener('submit', async e => {
  e.preventDefault();
  clearFieldErrors();
  const email    = document.getElementById('signinEmail').value.trim();
  const password = document.getElementById('signinPassword').value;

  let valid = true;
  if (!email)    { showFieldErr('signinEmail',    'Email is required');    valid=false; }
  if (!password) { showFieldErr('signinPassword', 'Password is required'); valid=false; }
  if (!valid) return;

  setBtnLoading('signinBtn', true);
  try {
    const res  = await fetch(`${API}/signin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (res.ok) {
      token       = data.token;
      currentUser = data.user;
      localStorage.setItem('pv_token', token);
      localStorage.setItem('pv_user',  JSON.stringify(currentUser));
      showApp();
    } else {
      showToast(data.message || 'Sign in failed', 'error');
    }
  } catch {
    showToast('Network error. Is the server running?', 'error');
  }
  setBtnLoading('signinBtn', false);
});

function logout() {
  token       = null;
  currentUser = null;
  localStorage.removeItem('pv_token');
  localStorage.removeItem('pv_user');
  showAuth();
}

function togglePass(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (inp.type === 'password') { inp.type = 'text';     btn.textContent = '🙈'; }
  else                          { inp.type = 'password'; btn.textContent = '👁'; }
}

/* ══════════════════════════════════════════════
   NAVIGATION
══════════════════════════════════════════════ */
function attachNav() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(name) {
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  document.querySelector(`.nav-item[data-tab="${name}"]`).classList.add('active');
  if (name === 'history')   loadHistory();
  if (name === 'dashboard') loadStats();
}

/* ══════════════════════════════════════════════
   DASHBOARD
══════════════════════════════════════════════ */
function selectPlatform(tile) {
  document.querySelectorAll('.platform-tile').forEach(t => t.classList.remove('selected'));
  tile.classList.add('selected');
  selectedPlatform = tile.dataset.platform;
  const form = document.getElementById('quick-form');
  form.style.display = 'flex';
  document.getElementById('quick-url').focus();
}

function quickVerify(e) {
  e.preventDefault();
  const url = document.getElementById('quick-url').value.trim();
  if (!url)              return showToast('Please enter a URL', 'error');
  if (!selectedPlatform) return showToast('Please select a platform', 'error');
  switchTab('verify');
  // Pre-fill verify form
  document.getElementById('v-platform').value = selectedPlatform;
  document.getElementById('v-url').value       = url;
  // Trigger verify
  runVerify(url, selectedPlatform);
}

/* ══════════════════════════════════════════════
   VERIFY
══════════════════════════════════════════════ */
function switchVerifyTab(name, btn) {
  document.querySelectorAll('.vtab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.vform').forEach(f => f.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(`vf-${name}`).classList.add('active');
  document.getElementById('v-result').style.display  = 'none';
  document.getElementById('v-loading').style.display = 'none';
}

function submitVerify(e, tab) {
  e.preventDefault();
  let url, platform;
  if (tab === 'standard') {
    platform = document.getElementById('v-platform').value;
    url      = document.getElementById('v-url').value.trim();
    if (!platform) return showToast('Please select a platform', 'error');
  } else {
    platform = 'linkedin';
    url      = document.getElementById('vl-url').value.trim();
  }
  if (!url) return showToast('Please enter a URL', 'error');
  runVerify(url, platform);
}

async function runVerify(url, platform) {
  currentVerifyUrl = url;
  document.getElementById('v-loading').style.display = 'flex';
  document.getElementById('v-result').style.display  = 'none';

  try {
    const res  = await fetch(`${API}/verify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      body:    JSON.stringify({ url, platform }),
    });
    const data = await res.json();
    if (!res.ok) { showToast(data.message || 'Verification failed', 'error'); return; }
    displayResult(data, url, platform);
    loadStats();
    loadHistory();
  } catch {
    showToast('Network error. Is the server running?', 'error');
  } finally {
    document.getElementById('v-loading').style.display = 'none';
  }
}

function displayResult(data, url, platform) {
  const isFake     = data.prediction === 'Fake';
  const card       = document.getElementById('v-result');
  const inner      = document.getElementById('v-result-inner');
  const actions    = document.getElementById('v-actions');
  const pct        = Math.round(data.confidence * 100);
  const username   = data.username || url.split('/').pop();

  // Build feature pills for active signals
  let featureHTML = '';
  if (data.features) {
    const active = Object.entries(data.features).filter(([,v]) => v > 0.3);
    if (active.length) {
      featureHTML = `<div class="feature-grid">` +
        active.map(([k]) => {
          const label = k.replace(/_/g,' ');
          return `<span class="feat-badge ${isFake?'warn':''}">${label}</span>`;
        }).join('') +
        `</div>`;
    }
  }

  inner.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
      <span style="font-size:2.2em;">${isFake ? '⚠️' : '✅'}</span>
      <h3 style="font-size:1.1em;color:var(--muted);font-weight:500;">Analysis Complete</h3>
    </div>
    <div class="result-verdict ${isFake ? 'fake':'real'}">
      ${isFake ? 'Fake Profile Detected' : 'Profile Appears Genuine'}
    </div>
    <div class="result-confidence">Confidence: <strong>${pct}%</strong></div>
    <div class="result-meta">
      <strong>Username:</strong> ${username}<br>
      <strong>Platform:</strong> ${platform}<br>
      <strong>URL:</strong> <a href="${url}" target="_blank" style="color:var(--brand);">${url}</a>
    </div>
    ${featureHTML}
  `;

  card.className = `result-card ${isFake ? 'is-fake' : 'is-real'}`;

  actions.innerHTML = isFake
    ? `<button class="btn btn-danger" onclick="openModal()">🚩 Report Profile</button>
       <button class="btn btn-ghost" onclick="openReportTab('${url}')">Open Report Tab</button>`
    : `<button class="btn btn-secondary" onclick="document.getElementById('v-result').style.display='none'">✕ Dismiss</button>`;

  card.style.display = 'block';
}

/* ══════════════════════════════════════════════
   HISTORY
══════════════════════════════════════════════ */
async function loadHistory() {
  const platform = document.getElementById('filter-platform')?.value || 'all';
  const result   = document.getElementById('filter-result')?.value   || 'all';

  let endpoint = `${API}/history`;
  const p = new URLSearchParams();
  if (platform !== 'all') p.append('platform', platform);
  if (result   !== 'all') p.append('result',   result);
  if (p.toString()) endpoint += '?' + p.toString();

  try {
    const res  = await fetch(endpoint, { headers: { 'Authorization': `Bearer ${token}` } });
    if (!res.ok) throw new Error();
    historyData = await res.json();
    renderHistory();
    renderRecent();
  } catch {
    document.getElementById('history-tbody').innerHTML =
      '<tr><td colspan="6" class="empty-row">Error loading history.</td></tr>';
  }
}

function renderHistory() {
  const tbody = document.getElementById('history-tbody');
  if (!historyData.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No verifications yet.</td></tr>';
    return;
  }
  tbody.innerHTML = historyData.map(row => {
    const username = (row.username || row.profileUrl.split('/').pop() || row.profileUrl).substring(0,30);
    const pct      = Math.round(row.confidence * 100);
    const date     = new Date(row.timestamp).toLocaleString();
    const id       = row._id;
    return `<tr>
      <td style="font-weight:600;">${username}</td>
      <td>${row.platform}</td>
      <td><span class="badge badge-${row.prediction === 'Real' ? 'real':'fake'}">${row.prediction}</span></td>
      <td>${pct}%</td>
      <td style="color:var(--muted);font-size:.85em;">${date}</td>
      <td style="display:flex;gap:6px;flex-wrap:wrap;">
        ${row.prediction === 'Fake'
          ? `<button class="btn btn-sm btn-danger"   onclick="reportFromHistory('${row.profileUrl}')">Report</button>`
          : ''}
        <button class="btn btn-sm btn-ghost" onclick="deleteRecord('${id}')">Delete</button>
      </td>
    </tr>`;
  }).join('');
}

function renderRecent() {
  const sec  = document.getElementById('recent-section');
  const list = document.getElementById('recent-list');
  if (!historyData.length) { sec.style.display = 'none'; return; }
  sec.style.display = 'block';
  list.innerHTML = historyData.slice(0,5).map(row => {
    const name = (row.username || row.profileUrl.split('/').pop()).substring(0,28);
    return `<div class="recent-item">
      <div>
        <div class="recent-url">${name}</div>
        <div class="recent-meta">${row.platform} · ${new Date(row.timestamp).toLocaleDateString()}</div>
      </div>
      <span class="badge badge-${row.prediction === 'Real' ? 'real':'fake'}">${row.prediction}</span>
    </div>`;
  }).join('');
}

async function deleteRecord(id) {
  if (!confirm('Delete this record?')) return;
  try {
    const res = await fetch(`${API}/history/${id}`, {
      method:  'DELETE',
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (res.ok) { showToast('Deleted', 'success'); loadHistory(); loadStats(); }
    else showToast('Delete failed', 'error');
  } catch { showToast('Network error', 'error'); }
}

function reportFromHistory(url) {
  currentVerifyUrl = url;
  openModal();
}

/* ══════════════════════════════════════════════
   STATS
══════════════════════════════════════════════ */
async function loadStats() {
  try {
    const res  = await fetch(`${API}/stats`, { headers: { 'Authorization': `Bearer ${token}` } });
    const data = await res.json();
    document.getElementById('stat-total').textContent = data.total || 0;
    document.getElementById('stat-fake').textContent  = data.fake  || 0;
    document.getElementById('stat-real').textContent  = (data.total - data.fake) || 0;
  } catch {
    // Fallback: compute from historyData
    const today = new Date().toDateString();
    const td    = historyData.filter(v => new Date(v.timestamp).toDateString() === today);
    document.getElementById('stat-total').textContent = td.length;
    document.getElementById('stat-fake').textContent  = td.filter(v => v.prediction === 'Fake').length;
    document.getElementById('stat-real').textContent  = td.filter(v => v.prediction === 'Real').length;
  }
}

/* ══════════════════════════════════════════════
   REPORT TAB
══════════════════════════════════════════════ */
function openReportTab(url) {
  currentVerifyUrl = url;
  document.getElementById('report-no-url').style.display = 'none';
  document.getElementById('report-form-area').style.display = 'block';
  document.getElementById('report-url-display').textContent = url;
  document.getElementById('rstep2').style.display = 'none';
  document.getElementById('rstep3').style.display = 'none';
  document.getElementById('rstep1').style.display = 'flex';
  switchTab('report');
}

function openPlatformReport() {
  if (currentVerifyUrl) window.open(currentVerifyUrl, '_blank');
  document.getElementById('rstep2').style.display = 'flex';
}

function handleReportedYes() {
  const msg = document.getElementById('report-final-msg');
  msg.innerHTML = `<p style="color:var(--success);font-weight:600;">✅ Thank you! Your report helps keep the platform safe.</p>`;
  document.getElementById('rstep3').style.display = 'flex';
  document.getElementById('rstep2').style.display = 'none';
}

function handleReportedNo() {
  const msg = document.getElementById('report-final-msg');
  msg.innerHTML = `<p style="color:var(--muted);">No problem. You can report the profile on the platform whenever you're ready.</p>`;
  document.getElementById('rstep3').style.display = 'flex';
  document.getElementById('rstep2').style.display = 'none';
}

/* ══════════════════════════════════════════════
   MODAL
══════════════════════════════════════════════ */
function openModal() {
  document.getElementById('report-modal').style.display = 'flex';
}
function closeModal() {
  document.getElementById('report-modal').style.display = 'none';
}
function confirmReport() {
  closeModal();
  showToast('Thank you for reporting!', 'success');
}

// Close modal on backdrop click
document.getElementById('report-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('report-modal')) closeModal();
});

/* ══════════════════════════════════════════════
   TOAST
══════════════════════════════════════════════ */
let toastTimer = null;
function showToast(msg, type = 'info') {
  const t = document.getElementById('toast');
  t.textContent  = msg;
  t.className    = `toast ${type}`;
  t.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.style.display = 'none'; }, 3500);
}
