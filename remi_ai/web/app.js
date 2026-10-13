/* Remi AI window: screens, X-ray viewer, finding panel and the Ask Remi assistant.
   All data and AI work happens in Python (remi_ai/api.py); this file only draws it. */

const ICON = {
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 10V2M5 5l3-3 3 3M3 9v4h10V9"/></svg>',
  search: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>',
  left: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 3 5 8l5 5"/></svg>',
  right: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m6 3 5 5-5 5"/></svg>',
  dots: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="3.5" cy="8" r="1.3"/><circle cx="8" cy="8" r="1.3"/><circle cx="12.5" cy="8" r="1.3"/></svg>',
  pen: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M11 2.5l2.5 2.5L6 12.5H3.5V10z"/></svg>',
  expand: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3h5v5M8 17H3v-5M17 3l-6 6M3 17l6-6"/></svg>',
  spark: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="9" cy="9" r="6"/><path d="M13.5 13.5 18 18"/><path d="M9 5.5v7M5.5 9h7" stroke-width="1.3"/></svg>',
  mic: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="5.5" y="1.5" width="5" height="8" rx="2.5"/><path d="M3 7.5a5 5 0 0 0 10 0M8 12.5V15"/></svg>',
  speaker: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2.5 6h2.5l3.5-3v10L5 10H2.5z"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.5a6 6 0 0 1 0 9"/></svg>',
  close: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  info: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6.5"/><path d="M8 7v4.5M8 4.6v.2"/></svg>',
  check: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" width="12" height="12"><path d="m3 8.5 3 3 7-7"/></svg>',
  plus: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 3v10M3 8h10"/></svg>'
};

const S = {
  screen: 'loading', info: null, doctor: null, patients: [], q: '',
  pid: null, pdata: null, xid: null, xray: null, f: 0, tab: 'finding',
  full: false, assist: false, menu: false, writing: false, busy: '', modal: null,
  bright: 50, contrast: 50, zoom: 100, overlay: 1, chat: [], listening: false, speak: false
};
const OVERLAY = ['OFF', 'LINES', 'AI MAP'];
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function fmtDate(iso, short) {
  if (!iso) return '—';
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number);
  return short ? `${d} ${MONTHS[m - 1]}` : `${String(d).padStart(2, '0')} ${MONTHS[m - 1]} ${y}`;
}
function monthsBetween(a, b) {
  const [ya, ma] = a.split('-').map(Number), [yb, mb] = b.split('-').map(Number);
  return Math.max(0, (yb - ya) * 12 + mb - ma);
}
function initials(name) { return name.split(/\s+/).filter(w => /^[A-Za-z]/.test(w) && !/^dr\.?$/i.test(w)).slice(0, 2).map(w => w[0].toUpperCase()).join('') || 'DR'; }

/* ---------- talking to Python ---------- */
function bridgeReady() {
  // The desktop window injects window.pywebview a moment after load; the dev server sets REMI_DEV.
  if (window.REMI_DEV || (window.pywebview && window.pywebview.api)) return Promise.resolve();
  return new Promise(r => window.addEventListener('pywebviewready', r, { once: true }));
}
async function call(name, ...args) {
  await bridgeReady();
  let res;
  if (window.pywebview && window.pywebview.api) res = await window.pywebview.api[name](...args);
  else res = await (await fetch('api/' + name, { method: 'POST', body: JSON.stringify(args) })).json();
  if (res && res.error) throw new Error(res.error);
  return res;
}
function toast(msg) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg;
  document.body.appendChild(t); setTimeout(() => t.remove(), 2600);
}
async function guard(fn) {
  try { return await fn(); } catch (e) { S.busy = ''; render(); toast(e.message); }
}

/* ---------- findings from a result ---------- */
const SCORE = {
  flat_foot: 'Flat foot model: 93.8% correct on 211 unseen X-rays.',
  heel_spur: 'Heel spur model: 97.5% correct on 595 unseen X-rays.',
  bunion: 'Bunion angles: within 3° of doctors for 85.0% (HVA) and 95.6% (IMA) of 160 test feet.'
};
function findingList(x) {
  const out = [], c = (x && x.results && x.results.checks) || {};
  if (c.flat_foot) out.push({ key: 'flat_foot', r: c.flat_foot });
  if (c.heel_spur) out.push({ key: 'heel_spur', r: c.heel_spur });
  if (c.bunion) c.bunion.feet.forEach((f, i) => out.push({ key: 'bunion', r: f, i }));
  return out;
}
function prevResult(fd) {
  const p = S.xray && S.xray.previous;
  if (!p) return null;
  const c = p.results.checks || {};
  if (fd.key === 'bunion') return c.bunion ? (c.bunion.feet.find(f => f.side === fd.r.side) || null) : null;
  return c[fd.key] || null;
}
function sevTag(sev, text) { return `<span class="tag ${sev === 'ok' ? 'green' : 'red'}">${esc(text)}</span>`; }

function describe(fd, x) {
  const r = fd.r, prev = prevResult(fd);
  if (fd.key === 'flat_foot') {
    const cat = r.category, measured = cat !== 'Not measured';
    const ang = measured ? r.angle : null, pAng = prev && prev.category !== 'Not measured' ? prev.angle : null;
    const diff = ang != null && pAng != null ? ang - pAng : null;
    return {
      title: cat === 'Not measured' ? 'Arch not measured' : cat,
      sub: `${r.left ? 'Left' : 'Right'} foot · arch height`,
      tags: (measured ? sevTag(r.severity, cat === 'Flat foot' ? 'Low arch' : cat === 'High arch' ? 'High arch' : 'Arch in normal range') : '<span class="tag">Check the X-ray framing</span>') + '<span class="tag">Side view</span>',
      bar: { label: 'Flat foot likelihood', pct: Math.round(r.p_flat * 100), calm: r.p_flat < 0.5 },
      score: SCORE.flat_foot,
      tiles: [
        { k: 'Heel bone angle', v: ang != null ? ang.toFixed(1) + '°' : '—', h: diff != null ? `${diff > 0 ? '+' : ''}${diff.toFixed(1)}° since ${fmtDate(S.xray.previous.date)}` : measured ? 'Lower means flatter' : 'Could not measure reliably', red: diff != null && diff < 0 },
        { k: 'Normal range', v: '17.7–30°', h: 'Under 17.7° is flat foot' },
        { k: 'Result', v: cat, sm: true, h: !measured ? 'Heel points look misplaced' : r.p_flat >= 0.5 === (cat === 'Flat foot') ? 'Both checks agree' : 'Angle and classifier differ' },
        { k: 'Foot', v: r.left ? 'Left' : 'Right', sm: true, h: 'Found automatically' }
      ],
      compare: prev && pAng != null && ang != null && r.points.L10 ? { region: flatRegion(r.points), prevRegion: flatRegion(prev.points), a: pAng.toFixed(1) + '°', b: ang != null ? ang.toFixed(1) + '°' : '—' } : null
    };
  }
  if (fd.key === 'heel_spur') {
    const p = r.probs, label = r.label;
    return {
      title: r.headline, sub: 'Heel · underside and back',
      tags: sevTag(r.severity, label === 'spur' ? 'Spur seen' : label === 'sever' ? "Child's heel irritation" : 'Heel looks clear') + '<span class="tag">Side view</span>',
      bar: { label: 'Heel spur likelihood', pct: Math.round(p.spur * 100), calm: label !== 'spur' },
      score: SCORE.heel_spur,
      tiles: [
        { k: 'Result', v: label === 'spur' ? 'Spur seen' : label === 'sever' ? "Sever's" : 'Not seen', sm: true, h: 'Marked on the X-ray' },
        { k: 'Spur', v: Math.round(p.spur * 100) + '%', h: 'Model likelihood', red: label === 'spur' },
        { k: 'Normal heel', v: Math.round(p.normal * 100) + '%', h: 'Model likelihood' },
        { k: "Child's heel (Sever's)", v: Math.round(p.sever * 100) + '%', h: 'Only in growing children' }
      ],
      compare: prev ? { region: boxRegion(r.box, 1.6), prevRegion: boxRegion(prev.box, 1.6), a: prev.headline.replace('Heel spur found', 'Spur seen'), b: r.headline.replace('Heel spur found', 'Spur seen') } : null
    };
  }
  const pHva = prev ? prev.hva : null;
  const grade = r.grade;
  return {
    title: grade === 'No bunion' ? 'No bunion' : 'Bunion', sub: `${r.side === 'left' ? 'Left' : 'Right'} foot · big toe joint`,
    tags: sevTag(r.severity, grade) + '<span class="tag">Top-down view</span>',
    bar: { label: 'Angle accuracy in testing (HVA within 3°)', pct: 85, calm: true },
    score: SCORE.bunion,
    tiles: [
      { k: 'Big toe angle (HVA)', v: r.hva.toFixed(1) + '°', h: pHva != null ? `${r.hva - pHva > 0 ? '+' : ''}${(r.hva - pHva).toFixed(1)}° since ${fmtDate(S.xray.previous.date)}` : 'Normal is under 15°', red: r.hva >= 15 },
      { k: 'Bone gap angle (IMA)', v: r.ima.toFixed(1) + '°', h: 'Normal is under 9°', red: r.ima >= 9 },
      { k: 'Grade', v: grade.replace(' bunion', ''), sm: true, h: 'Worse of the two angles' },
      { k: 'Foot', v: r.side === 'left' ? 'Left' : 'Right', sm: true, h: 'Found automatically' }
    ],
    compare: prev ? { region: ptsRegion(r.points), prevRegion: ptsRegion(prev.points), a: prev.hva.toFixed(1) + '°', b: r.hva.toFixed(1) + '°' } : null
  };
}
function boxRegion(b, pad) {
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2, s = Math.max(b[2] - b[0], b[3] - b[1]) * pad;
  return [cx - s / 2, cy - s / 2, s, s];
}
function flatRegion(p) {
  if (!p || !p.L10) return [0, 0, 1, 1];
  const [x1, y1] = p.L10, [x2, y2] = p.L11;
  return boxRegion([Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)], 1.8);
}
function ptsRegion(pts) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  return boxRegion([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], 1.3);
}

/* ---------- overlays on the X-ray ---------- */
function heatImage(grid, cmapAlpha = 0.85) {
  const h = grid.length, w = grid[0].length;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d'), d = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = Math.max(0, Math.min(1, grid[y][x])), i = (y * w + x) * 4;
    d.data[i] = 255; d.data[i + 1] = Math.round(70 + 170 * v * v); d.data[i + 2] = Math.round(60 * v);
    d.data[i + 3] = Math.round(255 * cmapAlpha * Math.max(0, v - 0.15) / 0.85);
  }
  ctx.putImageData(d, 0, 0);
  return c.toDataURL();
}
function pill(x, y, text, u, red) {
  const fs = 15 * u, w = (text.length * 0.62 + 1.4) * fs, h = fs * 1.9;
  return `<rect class="ov-pill" x="${x}" y="${y - h / 2}" width="${w}" height="${h}" rx="${7 * u}"/>
    <text class="ov-label${red ? ' red' : ''}" x="${x + 0.7 * fs}" y="${y + fs * 0.36}" font-size="${fs}">${esc(text)}</text>`;
}
function overlaySvg(x, fd, u) {
  if (!fd || S.overlay === 0) return '';
  const r = fd.r;
  let out = '';
  if (S.overlay === 2) {
    if (fd.key === 'flat_foot' && r.heatmap && r.heatmap.length) out += `<image href="${heatImage(r.heatmap)}" x="0" y="0" width="${x.width}" height="${x.height}" preserveAspectRatio="none" style="mix-blend-mode:screen"/>`;
    if (fd.key === 'heel_spur' && r.heatmap) { const b = r.crop_box; out += `<image href="${heatImage(r.heatmap)}" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" preserveAspectRatio="none" style="mix-blend-mode:screen"/>`; }
  }
  if (fd.key === 'flat_foot' && r.points && r.points.L10 && r.category !== 'Not measured') {
    const [x1, y1] = r.points.L10, [x2, y2] = r.points.L11;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    const ex = x1 + dx * 1.25, ey = y1 + dy * 1.25, dir = Math.sign(dx) || 1;
    out += `<line class="ov-dash" x1="${x1 - dir * len * 0.3}" y1="${y1}" x2="${x1 + dir * len * 1.6}" y2="${y1}"/>
      <line class="ov-line" x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}"/>
      <circle class="ov-dot" cx="${x1}" cy="${y1}" r="${4 * u}"/><circle class="ov-dot" cx="${x2}" cy="${y2}" r="${4 * u}"/>
      ${pill(Math.min(x1, x2) + Math.abs(dx) * 0.25, Math.max(y1, y2) + 30 * u, `Arch ${r.angle != null ? r.angle.toFixed(1) + '°' : '—'}`, u)}`;
  }
  if (fd.key === 'heel_spur') {
    const [a, b, c, d] = r.box, cx = (a + c) / 2, cy = (b + d) / 2, rx = (c - a) / 2 * 1.05, ry = (d - b) / 2 * 1.05;
    const spur = r.label === 'spur';
    out += `<ellipse class="${spur ? 'ov-red' : 'ov-dash'}" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"/>
      ${pill(cx - rx, d + 26 * u, spur ? 'Heel spur' : r.label === 'sever' ? "Sever's" : 'No spur seen', u, spur)}`;
  }
  if (fd.key === 'bunion') {
    const p = r.points, ext = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
    const seg = (i, cls) => { const s = ext(p[i + 1], p[i], 1.15), e = ext(p[i], p[i + 1], 1.15); return `<line class="${cls}" x1="${s[0]}" y1="${s[1]}" x2="${e[0]}" y2="${e[1]}"/>`; };
    out += seg(0, 'ov-line') + seg(2, 'ov-line') + seg(4, 'ov-dash');
    p.forEach(q => { out += `<circle class="ov-dot" cx="${q[0]}" cy="${q[1]}" r="${3.5 * u}"/>`; });
    const sideX = r.side === 'left' ? 1 : -1;
    const hx = Math.min(p[0][0], p[1][0], p[2][0]) - 150 * u, hy = p[1][1];
    out += pill(sideX > 0 ? Math.max(p[0][0], p[1][0]) + 30 * u : hx, hy, `HVA ${r.hva.toFixed(1)}°`, u, r.hva >= 15);
    out += pill(Math.max(p[3][0], p[5][0]) + 30 * u, (p[3][1] + p[5][1]) / 2, `IMA ${r.ima.toFixed(1)}°`, u, r.ima >= 9);
  }
  return out;
}
function xraySvg(x, fd) {
  const u = Math.max(x.width, x.height) / 900;
  const filt = `filter: brightness(${(0.5 + S.bright / 100).toFixed(2)}) contrast(${(0.5 + S.contrast / 100).toFixed(2)});`;
  return `<svg class="xr" viewBox="0 0 ${x.width} ${x.height}" preserveAspectRatio="xMidYMid meet" style="transform: scale(${S.zoom / 100})" role="img" aria-label="${esc(x.view_label)} X-ray">
    <image href="${x.image}" x="0" y="0" width="${x.width}" height="${x.height}" preserveAspectRatio="none" style="${filt}"/>
    ${overlaySvg(x, fd, u)}
  </svg>`;
}
function lens(image, w, h, region) {
  return `<svg viewBox="${region.map(v => v.toFixed(1)).join(' ')}" preserveAspectRatio="xMidYMid slice"><rect x="-99999" y="-99999" width="199999" height="199999" fill="#000"/><image href="${image}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none"/></svg>`;
}

/* ---------- screens ---------- */
function introScreen() {
  return `<div class="intro"><div class="intro-inner">
    <div><div class="eyebrow">Remi AI · foot X-ray assistant</div>
      <h1>Reads the foot X-ray, so you can spend the time on the patient.</h1></div>
    <p class="lead">Remi checks each foot X-ray with the AI model that fits it and puts the measurements next to the image. You stay in charge: every number is a suggestion you can accept, change or ignore.</p>
    <div class="steps">
      <div class="step"><div class="n">1</div><div><b>Sign in with your doctor ID</b><span>Your patients and notes stay on this computer.</span></div></div>
      <div class="step"><div class="n">2</div><div><b>Open a patient or add a new one</b><span>Each patient has their own record: first visit, medical history and every X-ray over time.</span></div></div>
      <div class="step"><div class="n">3</div><div><b>Add an X-ray</b><span>Remi first checks what kind of X-ray it is. A side view gets the flat foot and heel spur checks. A top-down view gets the bunion angles. Anything that isn't a foot X-ray is skipped.</span></div></div>
      <div class="step"><div class="n">4</div><div><b>Review the findings</b><span>Compare with the last visit, add notes, refer the patient or export a report. Ask Remi by voice or text if you're unsure what to do next.</span></div></div>
    </div>
    <div class="foot"><button class="btn solid" data-act="introDone">Get started</button><span>Remi helps with screening. It does not replace a doctor's diagnosis.</span></div>
  </div></div>`;
}
function loginScreen() {
  const first = !S.info.has_doctors;
  return `<div class="login"><form class="login-card" id="${first ? 'setupForm' : 'loginForm'}">
    <div><div class="eyebrow">Remi AI</div><h2>${first ? 'Create your doctor account' : 'Sign in'}</h2></div>
    ${first ? `<div class="field"><label for="docName">Your name</label><input id="docName" placeholder="Dr. Kavya Rao" required></div>
      <div class="field"><label for="clinic">Clinic</label><input id="clinic" placeholder="Optional"></div>` : ''}
    <div class="field"><label for="docId">Doctor ID</label><input id="docId" autocomplete="username" required placeholder="e.g. DR-2207"></div>
    <div class="field"><label for="pw">Password</label><input id="pw" type="password" autocomplete="${first ? 'new-password' : 'current-password'}" required placeholder="${first ? 'At least 6 characters' : ''}"></div>
    <div class="err" id="loginErr"></div>
    <button class="btn solid" type="submit">${first ? 'Create account' : 'Sign in'}</button>
    <small>${first ? 'This is the first sign-in on this computer. Patient files are kept here, in your own folder.' : 'Patient files stay on this computer.'}</small>
  </form></div>`;
}
function findingTags(list) {
  if (!list.length) return '<span class="tag">No X-rays yet</span>';
  const bad = list.filter(f => f[2] !== 'ok');
  if (!bad.length) return '<span class="tag green">No findings</span>';
  return bad.map(f => `<span class="tag red">${esc(f[0])}</span>`).join('');
}
function listScreen() {
  const rows = S.patients;
  const models = S.info.models, missing = Object.keys(models).filter(k => !models[k]);
  return `<div class="list">
    <div class="list-head">
      <div><div class="eyebrow">Signed in as ${esc(S.doctor.name)}</div><h2>Patients</h2></div>
      <div class="list-tools">
        <label class="search">${ICON.search}<input id="listSearch" placeholder="Search name or ID" value="${esc(S.q)}"></label>
        <button class="btn solid" data-act="newPatient">${ICON.plus}New patient</button>
        <button class="btn" data-act="assist">${ICON.spark}Ask Remi</button>
        <button class="avatar" data-act="logout" title="${esc(S.doctor.name)} · click to sign out">${initials(S.doctor.name)}</button>
      </div>
    </div>
    ${missing.length ? `<div class="card" style="margin-bottom:14px"><div class="skipnote">${ICON.info}<span>Some AI model files are missing on this computer: ${missing.map(k => esc((S.info.checks[k] || { name: 'X-ray check' }).name)).join(', ')}. Those checks will be skipped. ${missing.includes('view_check') ? 'Without the X-ray check you will be asked what kind of X-ray each upload is.' : ''}</span></div></div>` : ''}
    ${rows.length || S.q ? `<div class="table-wrap"><table>
      <thead><tr><th>Patient</th><th>ID</th><th>Last visit</th><th>X-rays</th><th>Latest findings</th></tr></thead>
      <tbody>${rows.map(p => `<tr data-open="${esc(p.patient_id)}" tabindex="0">
        <td><div class="who"><b>${esc(p.name)}</b><span>${[p.age, p.sex].filter(Boolean).map(esc).join(' · ') || '—'}${p.unreviewed ? ` · ${p.unreviewed} to review` : ''}</span></div></td>
        <td class="num">${esc(p.patient_id)}</td><td class="num">${fmtDate(p.last_visit || p.created)}</td><td class="num">${p.n_xrays}</td>
        <td>${findingTags(p.findings)}</td></tr>`).join('') || '<tr><td colspan="5" style="color:var(--muted)">No patient matches that search.</td></tr>'}</tbody>
    </table></div>` : `<div class="table-wrap"><div class="list-empty"><b style="font-weight:500;color:var(--fg);font-size:15px">No patients yet</b><span>Add your first patient to start their record, then add their X-rays.</span><button class="btn solid" data-act="newPatient">${ICON.plus}New patient</button></div></div>`}
    ${S.assist ? assistPanel(true) : ''}
  </div>`;
}

function allXrays() {
  const out = [];
  if (!S.pdata) return out;
  S.pdata.visits.forEach(v => v.xrays.forEach(x => out.push({ ...x, date: v.date })));
  return out;
}
function chips() {
  const c = [['BRT', S.bright, 'bright'], ['CON', S.contrast, 'contrast'], ['ZM', S.zoom + '%', 'zoom'], ['OVL', OVERLAY[S.overlay], 'overlay']];
  return `<div class="chips" aria-label="Image settings">${c.map(([k, v, a]) => `<button data-chip="${a}" class="${a === 'overlay' && !S.overlay ? 'off' : ''}" title="Click to change"><span>${k}</span><b class="num" style="font-weight:400">${v}</b></button>`).join('')}</div>`;
}
function viewPicker(title) {
  return `<div class="card viewpick"><div class="f-title">${title}</div>
    <div class="f-sub">${S.info.models.view_check ? 'The X-ray check was not sure about this one.' : 'The X-ray check model is not installed.'} Pick the view so Remi runs the right checks.</div>
    <div class="row3"><button data-view="side">Side view<small>Flat foot, heel spur</small></button><button data-view="top_down">Top-down<small>Bunion angles</small></button><button data-view="not_foot">Not a foot<small>No checks</small></button></div></div>`;
}
function skippedLines(x) {
  const sk = (x.results && x.results.skipped) || {};
  const names = Object.keys(sk).map(k => `${S.info.checks[k].name}: ${sk[k].replace(/^Not run: /, 'not run, ')}`);
  return names.length ? `<div class="skipnote">${ICON.info}<span>${names.map(esc).join('. ')}.</span></div>` : '';
}
function findingPanel() {
  const x = S.xray;
  if (!x) return `<div class="card"><div class="f-title">No X-ray yet</div><div class="f-sub">Add an X-ray to see findings</div></div>
    <p class="empty-note">Drop an X-ray on the left or press Add X-ray. Remi checks what kind of X-ray it is and runs only the models that fit.</p>`;
  if (x.view === 'unsure' || S.menu === 'view') return viewPicker(x.view === 'unsure' ? 'What kind of X-ray is this?' : 'Change the X-ray type');
  const list = findingList(x);
  if (!list.length) {
    const why = x.view === 'not_foot' ? 'This is not a foot X-ray, so no foot checks ran.' : x.view === 'oblique' ? 'This is an angled view. The models need a straight side or top-down view.' : 'No check could read this X-ray.';
    return `<div class="card"><div class="f-title">No checks ran</div><div class="f-sub">X-ray check: ${esc(x.view_label)}${x.view_source === 'doctor' ? ' (set by you)' : ''}</div>
      <div class="f-tags"><span class="tag">${esc(x.view_label)}</span><span class="tag">Skipped</span></div></div>
      <p class="empty-note">${why} If this is wrong, use the menu above to change the X-ray type.</p>${skippedLines(x)}`;
  }
  S.f = Math.min(S.f, list.length - 1);
  const fd = list[S.f], c = describe(fd, x), cmp = c.compare, pv = x.previous;
  const on = Math.round(c.bar.pct / 100 * 40);
  return `
    <div class="card">
      <div class="f-title">${esc(c.title)}</div><div class="f-sub">${esc(c.sub)}</div>
      <div class="f-tags">${c.tags}${x.reviewed ? `<span class="tag green">${ICON.check}&nbsp;Reviewed</span>` : ''}</div>
    </div>
    <div class="card">
      <div class="conf-head"><span>${esc(c.bar.label)}</span><b class="num">${c.bar.pct}<small>%</small></b></div>
      <div class="segs${c.bar.calm ? ' calm' : ''}">${Array.from({ length: 40 }, (_, i) => `<i class="${i < on ? 'on' : ''}"></i>`).join('')}</div>
      <div class="scale"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
      <div class="note-line">${esc(c.score)}</div>
    </div>
    <div class="tiles">${c.tiles.map(t => `<div class="tile"><span class="k">${esc(t.k)}</span><span class="v${t.sm ? ' sm' : ''}">${esc(t.v)}</span><span class="h${t.red ? ' red' : ''}">${esc(t.h)}</span></div>`).join('')}</div>
    ${cmp ? `<div class="card">
      <div class="compare-head"><span>Previous comparison</span><b class="num">${monthsBetween(pv.date, x.date)} months</b></div>
      <div class="compare">
        <div class="cmp"><span class="date">${fmtDate(pv.date)}</span><div class="lens">${lens(pv.image, pv.width, pv.height, cmp.prevRegion)}</div><div class="row"><span class="num">${esc(cmp.a)}</span><b>PREV</b></div></div>
        <div class="cmp"><span class="date">${fmtDate(x.date)}</span><div class="lens">${lens(x.image, x.width, x.height, cmp.region)}</div><div class="row"><span class="num">${esc(cmp.b)}</span><b class="cur">CURRENT</b></div></div>
      </div></div>` : ''}
    ${skippedLines(x)}
    ${S.writing ? `<div class="card"><textarea class="notes" id="noteBox" placeholder="Note for today's visit"></textarea>
      <div class="actions" style="margin-top:8px"><button class="btn" data-act="cancelNote">Cancel</button><button class="btn solid" data-act="saveNote">Save note</button></div></div>` : ''}
    <div class="actions">
      <button class="btn" data-act="note">${ICON.pen}Add Notes</button>
      <button class="btn solid" data-act="refer">Refer Patient</button>
    </div>`;
}
function recordPanel() {
  const p = S.pdata.patient, h = S.pdata.history;
  const histItems = Object.keys(h).length ? Object.entries(h).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('') : '<dt>—</dt><dd style="color:var(--muted)">Nothing recorded yet</dd>';
  return `<div class="card rec"><h4>Patient</h4><dl class="kv">
      <dt>Name</dt><dd>${esc(p.name)}</dd><dt>Age / sex</dt><dd>${[p.age, p.sex].filter(Boolean).map(esc).join(' · ') || '—'}</dd>
      <dt>Patient ID</dt><dd class="num">${esc(p.patient_id)}</dd><dt>Phone</dt><dd>${esc(p.phone) || '—'}</dd></dl></div>
    <div class="card rec"><h4>Initial history</h4><dl class="kv">
      <dt>Complaint</dt><dd>${esc(p.complaint) || '—'}</dd><dt>First seen</dt><dd class="num">${fmtDate(p.created)}</dd><dt>Referred by</dt><dd>${esc(p.referred_by) || '—'}</dd></dl></div>
    <div class="card rec"><h4>Medical history</h4><dl class="kv">${histItems}</dl></div>
    <div class="card rec"><h4>Visits</h4><div class="timeline">
      ${S.pdata.visits.map(v => `<div class="tl"><span class="num">${fmtDate(v.date)}</span><div><b style="font-weight:500">${esc(v.label)}</b>
        ${v.xrays.map(x => `<p>${esc(x.view_label)}: ${x.findings.length ? x.findings.map(f => esc(f[0])).join(', ') : 'no checks ran'}</p>`).join('')}
        ${v.note ? `<p style="white-space:pre-line">${esc(v.note)}</p>` : ''}${!v.xrays.length && !v.note ? '<p>No X-rays or notes</p>' : ''}</div></div>`).join('')}
    </div></div>
    <div class="actions"><button class="btn" data-act="openFolder">Open patient folder</button><button class="btn solid" data-act="editPatient">Edit details</button></div>`;
}
function patientScreen() {
  const p = S.pdata.patient, xs = allXrays(), x = S.xray;
  const list = x ? findingList(x) : [];
  const fd = list[Math.min(S.f, list.length - 1)];
  const stage = x
    ? (x.view === 'not_foot' ? `<div style="position:relative;width:100%;height:100%">${xraySvg(x, null)}</div>` : xraySvg(x, fd))
    : `<div class="drop" id="drop"><div class="big">Add ${esc(p.name.split(' ')[0])}'s first X-ray</div><span>Drag an X-ray file here, or</span><button class="btn solid" data-act="addXray">${ICON.plus}Choose X-ray</button><span style="font-size:11.5px">JPG, PNG or TIFF</span></div>`;
  const nFind = list.length;
  return `<div class="viewer${S.full ? ' full' : ''}" id="viewer">
    <div class="grid"></div>
    <div class="xray-stage">${stage}</div>
    <div class="topleft">
      <button class="icon-btn" data-act="back" aria-label="Back to patients">${ICON.left}</button>
      <div class="crumb"><b>${esc(p.name)}</b><span class="num">${esc(p.patient_id)}</span></div>
      <div class="views" role="tablist">${xs.map(v => `<button class="${v.id === S.xid ? 'on' : ''}" data-xray="${v.id}">${esc(v.view === 'unsure' ? 'Needs view' : v.view_label)} · ${fmtDate(v.date, true)}</button>`).join('')}<button class="add" data-act="addXray">+ Add X-ray</button></div>
    </div>
    ${x ? chips() : ''}
    <div class="side">
      <div class="toolbar">
        <button class="btn" data-act="export">${ICON.export}Export</button>
        <button class="btn solid" data-act="save" ${x && x.view !== 'unsure' ? '' : 'disabled style="opacity:.5"'}>${x && x.reviewed ? 'Saved' : 'Save'}</button>
        <label class="search">${ICON.search}<input id="topSearch" placeholder="Search"></label>
        <button class="avatar" data-act="logout" title="${esc(S.doctor.name)} · click to sign out">${initials(S.doctor.name)}</button>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div class="tabs"><button class="${S.tab === 'finding' ? 'on' : ''}" data-tab="finding">${nFind ? `Finding ${String(S.f + 1).padStart(2, '0')}` : 'Findings'}</button><button class="${S.tab === 'record' ? 'on' : ''}" data-tab="record">Record</button></div>
          ${S.tab === 'finding' && x ? `<div class="navs">${nFind > 1 ? `<button class="icon-btn" data-act="prevF" aria-label="Previous finding">${ICON.left}</button><button class="icon-btn" data-act="nextF" aria-label="Next finding">${ICON.right}</button>` : ''}<button class="icon-btn" data-act="menu" aria-label="More">${ICON.dots}</button></div>` : ''}
        </div>
        ${S.menu === true ? `<div class="menu"><button data-act="changeView">Change X-ray type</button><button data-act="unreview">${x && x.reviewed ? 'Mark as not reviewed' : 'Mark as reviewed'}</button><button class="danger" data-act="deleteXray">Delete this X-ray</button></div>` : ''}
        ${S.tab === 'finding' ? findingPanel() : recordPanel()}
      </div>
    </div>
    <div class="fabs">
      <button class="fab${S.full ? ' on' : ''}" data-act="full" aria-label="Full view" title="Full view">${ICON.expand}</button>
      <button class="fab${S.assist ? ' on' : ''}" data-act="assist" aria-label="Ask Remi" title="Ask Remi">${ICON.spark}</button>
    </div>
    ${S.assist ? assistPanel(false) : ''}
    ${S.busy ? `<div class="busy"><div class="busy-box"><div class="spinner"></div><span>${esc(S.busy)}</span></div></div>` : ''}
  </div>`;
}

/* ---------- modals ---------- */
function patientForm(p, h) {
  const items = [...new Set([...S.info.history_items, ...Object.keys(h || {})])];
  return `<div class="modal-back"><form class="modal" id="patientForm">
    <h3>${p ? 'Edit patient' : 'New patient'}</h3>
    <p>The patient ID is your clinic's own number. This patient's X-rays are kept in their own folder on this computer.</p>
    <div class="form-grid">
      <div class="field"><label for="f_pid">Patient ID</label><input id="f_pid" required value="${esc(p ? p.patient_id : S.modal.nextId)}" ${p ? 'disabled' : ''}></div>
      <div class="field"><label for="f_name">Full name</label><input id="f_name" required value="${esc(p ? p.name : '')}"></div>
      <div class="field"><label for="f_age">Age</label><input id="f_age" inputmode="numeric" value="${esc(p ? p.age ?? '' : '')}"></div>
      <div class="field"><label for="f_sex">Sex</label><select id="f_sex">${['', 'Female', 'Male', 'Other'].map(s => `<option ${p && p.sex === s ? 'selected' : ''}>${s}</option>`).join('')}</select></div>
      <div class="field"><label for="f_phone">Phone</label><input id="f_phone" value="${esc(p ? p.phone : '')}"></div>
      <div class="field"><label for="f_ref">Referred by</label><input id="f_ref" value="${esc(p ? p.referred_by : '')}"></div>
      <div class="field wide"><label for="f_complaint">Main complaint</label><textarea id="f_complaint" placeholder="Why they came in, e.g. heel pain for 4 months, worst in the morning">${esc(p ? p.complaint : '')}</textarea></div>
      <div class="section-label wide">Medical history</div>
      ${items.map((it, i) => `<div class="field"><label for="h_${i}">${esc(it)}</label><input id="h_${i}" data-hist="${esc(it)}" value="${esc((h || {})[it] || '')}" placeholder="—"></div>`).join('')}
    </div>
    <div class="err" id="formErr"></div>
    <div class="actions"><button type="button" class="btn" data-act="closeModal">Cancel</button><button class="btn solid" type="submit">${p ? 'Save changes' : 'Add patient'}</button></div>
  </form></div>`;
}
function referForm() {
  const p = S.pdata.patient;
  return `<div class="modal-back"><form class="modal" id="referForm">
    <h3>Refer ${esc(p.name)}</h3>
    <p>Remi writes a referral letter with the patient's details, history and today's X-ray findings, then opens it so you can print or save it.</p>
    <div class="field"><label for="r_to">Refer to</label><input id="r_to" placeholder="e.g. Dr. S. Nair, Orthopaedics, City Hospital"></div>
    <div class="field"><label for="r_reason">Reason</label><textarea id="r_reason" placeholder="e.g. Moderate bunion with pain in shoes, for surgical opinion"></textarea></div>
    <div class="actions"><button type="button" class="btn" data-act="closeModal">Cancel</button><button class="btn solid" type="submit">Write referral</button></div>
  </form></div>`;
}
function confirmForm(text, act) {
  return `<div class="modal-back"><div class="modal"><h3>${esc(text)}</h3><p>This removes the X-ray file and its results from the patient's record.</p>
    <div class="actions"><button class="btn" data-act="closeModal">Cancel</button><button class="btn solid" data-act="${act}">Delete</button></div></div></div>`;
}

/* ---------- Ask Remi ---------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recog = null;
function assistPanel(global) {
  if (!S.chat.length) S.chat.push({ me: false, t: `Hi ${S.doctor.name.split(' ').slice(0, 2).join(' ')}. I can guide you through Remi step by step. ${nextStep()}` });
  const sugg = S.screen === 'patient'
    ? ['What do I do next?', 'Explain this finding', 'Why was a check skipped?', 'Show the record']
    : ['What do I do next?', 'Add a new patient', 'How accurate are the models?'];
  return `<div class="assist${global ? ' global' : ''}" role="dialog" aria-label="Ask Remi">
    <div class="assist-head"><div><b>Ask Remi</b><span>Voice or text · guides you step by step</span></div>
      <div style="display:flex;gap:6px"><button class="icon-btn${S.speak ? '' : ' dim'}" data-act="speak" aria-label="Read answers aloud" title="Read answers aloud">${ICON.speaker}</button><button class="icon-btn" data-act="closeAssist" aria-label="Close">${ICON.close}</button></div></div>
    <div class="msgs" id="msgs">${S.chat.map(m => `<div class="msg ${m.me ? 'me' : 'bot'}">${esc(m.t)}</div>`).join('')}</div>
    <div class="sugg">${sugg.map(s => `<button data-say="${s}">${s}</button>`).join('')}</div>
    <form class="composer" id="chatForm"><button type="button" class="mic${S.listening ? ' live' : ''}" data-act="mic" aria-label="Speak">${ICON.mic}</button>
      <input id="chatIn" placeholder="${S.listening ? 'Listening…' : 'Type a question or command'}" autocomplete="off"><button class="btn solid" type="submit">Send</button></form>
  </div>`;
}
function nextStep() {
  if (S.screen === 'list') return S.patients.length ? 'Open a patient from the list, or say "new patient" to add one.' : 'Start by adding your first patient: say "new patient" or press New patient.';
  if (S.screen === 'patient') {
    const x = S.xray;
    if (!x) return 'Add an X-ray for this patient: press Add X-ray or drop the file on the left.';
    if (x.view === 'unsure') return 'Tell me what kind of X-ray this is using the three buttons on the right, so I run the right checks.';
    if (!findingList(x).length) return 'No checks ran on this X-ray. Add a side or top-down foot X-ray, or change the X-ray type from the menu.';
    if (!x.reviewed) return 'Look at the lines on the X-ray and the numbers on the right. If you agree, press Save to mark it reviewed. Then add a note or refer the patient if needed.';
    return 'This X-ray is reviewed. You can add a note, refer the patient, or press Export for a report.';
  }
  return '';
}
function explainFinding() {
  const x = S.xray; if (!x) return 'There is no X-ray open yet.';
  const list = findingList(x); if (!list.length) return 'No checks ran on this X-ray, so there is nothing to explain.';
  const fd = list[S.f], r = fd.r;
  if (fd.key === 'flat_foot') return `The heel bone angle is the slope of the heel bone against the ground, seen from the side. Between 17.7° and 30° is normal; lower means a flatter arch. ${r.category === 'Not measured' ? 'On this X-ray Remi could not place the heel points reliably, so it gives no angle. Check that it is a standing side view of the whole foot.' : `This foot measures ${r.angle.toFixed(1)}°, so Remi reads it as: ${r.category.toLowerCase()}.`}`;
  if (fd.key === 'heel_spur') return `Remi finds the heel bone, then looks for a bony spur under or behind it. Here it says: ${r.headline.toLowerCase()} (${Math.round(r.probs[r.label] * 100)}% likely). The circle on the X-ray marks the heel area it checked.`;
  return `HVA is how far the big toe leans towards the other toes (normal under 15°). IMA is the gap angle between the 1st and 2nd foot bones (normal under 9°). This foot has HVA ${r.hva.toFixed(1)}° and IMA ${r.ima.toFixed(1)}°, so: ${r.grade.toLowerCase()}.`;
}
function answer(q) {
  const s = q.toLowerCase().trim();
  const onPatient = S.screen === 'patient';
  let m;
  if (/(new|add|create).*patient/.test(s)) { openNewPatient(); return 'Opened the new patient form. Fill in the ID and name; the medical history can be added later.'; }
  if ((m = s.match(/^(open|show|find)\s+(patient\s+)?(.+)$/)) && !/(record|history|side|top|finding|folder|report)/.test(s)) {
    const term = m[3].trim(); const p = S.patients.find(p => p.name.toLowerCase().includes(term) || p.patient_id.toLowerCase() === term);
    if (p) { openPatient(p.patient_id); return `Opening ${p.name}.`; }
    return `I couldn't find a patient called "${m[3]}". Check the spelling or the patient ID.`;
  }
  if (/(patients|patient list|go back|^back)/.test(s)) { goList(); return 'Back to the patient list.'; }
  if (/accura|how good|trust/.test(s)) return 'In testing: flat foot 93.8% correct on 211 unseen X-rays; heel spur 97.5% on 595; bunion angles within 3° of doctors for 85% (HVA) and 95.6% (IMA) of 160 feet. Always check the result yourself.';
  if (/explain|what does|mean|angle|hva|ima/.test(s)) return onPatient ? explainFinding() : 'Open a patient and an X-ray, then ask me to explain a finding.';
  if (/finding/.test(s) && /next/.test(s) && onPatient) { S.f++; S.tab = 'finding'; return 'Next finding.'; }
  if (/(what|next|help|guide|how do i|start)/.test(s)) return nextStep();
  if (!onPatient) return 'Open a patient first. Then I can explain findings, switch X-rays, add notes and write referrals.';
  if (/(add|upload|new).*x-?ray|upload/.test(s)) { $('#fileIn').click(); return 'Choose the X-ray file. I will check what kind of X-ray it is and run only the models that fit.'; }
  if (/skip|not run|why.*(no|not)/.test(s)) { const x = S.xray; const sk = x && x.results.skipped ? Object.entries(x.results.skipped) : []; return sk.length ? sk.map(([k, v]) => `${S.info.checks[k].name}: ${v.replace(/^Not run: /, 'not run because it ')}`).join('. ') + '. Each model only runs on the X-ray view it was trained on.' : 'Nothing was skipped on this X-ray.'; }
  if (/side/.test(s) || /top/.test(s)) { const want = /side/.test(s) ? 'side' : 'top_down'; const x = allXrays().find(v => v.view === want); if (x) { openXray(x.id); return `Opened the ${want === 'side' ? 'side' : 'top-down'} view.`; } return `This patient has no ${want === 'side' ? 'side' : 'top-down'} X-ray yet.`; }
  if (/record|history/.test(s)) { S.tab = 'record'; return 'The record is open on the right: complaint, medical history and every visit.'; }
  if (/finding/.test(s)) { S.tab = 'finding'; return 'Showing the findings.'; }
  if ((m = q.match(/^\s*note[:,]?\s+(.+)/i))) { guard(async () => { await call('add_note', S.pid, m[1]); await reloadPatient(); }); return 'Saved that note to today\'s visit.'; }
  if (/save|review/.test(s)) { doAct('save'); return 'Marked this X-ray as reviewed.'; }
  if (/export|report/.test(s)) { doAct('export'); return 'Writing the report. It opens in your browser, where you can print it or save it as PDF.'; }
  if (/refer/.test(s)) { doAct('refer'); return 'Opened the referral form.'; }
  if (/zoom in/.test(s)) { S.zoom = Math.min(200, S.zoom + 25); return 'Zoomed in.'; }
  if (/zoom out/.test(s)) { S.zoom = Math.max(50, S.zoom - 25); return 'Zoomed out.'; }
  if (/bright/.test(s)) { S.bright = Math.min(100, S.bright + 10); return 'Brighter.'; }
  if (/dark/.test(s)) { S.bright = Math.max(0, S.bright - 10); return 'Darker.'; }
  if (/heat|map|focus/.test(s)) { S.overlay = 2; return 'Showing where the AI looked (the warm areas).'; }
  if (/hide.*line|lines off/.test(s)) { S.overlay = 0; return 'Lines hidden.'; }
  if (/line/.test(s)) { S.overlay = 1; return 'Lines shown.'; }
  if (/full|hide panel/.test(s)) { S.full = true; return 'Full view. Press the expand button to bring the panel back.'; }
  return 'I can: explain this finding, say why a check was skipped, open the side or top view, add a note (say "note: ..."), save, export a report, refer the patient, zoom, or show where the AI looked.';
}
function say(text) {
  S.chat.push({ me: true, t: text });
  const a = answer(text);
  S.chat.push({ me: false, t: a });
  if (S.speak && window.speechSynthesis) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(a)); }
  render();
  const i = $('#chatIn'); if (i) i.focus();
}
function toggleMic() {
  if (!SR) { toast('Voice input is not available on this computer. Type your question instead.'); return; }
  if (S.listening && recog) { recog.stop(); return; }
  recog = new SR(); recog.lang = 'en-IN'; recog.interimResults = false;
  recog.onresult = (e) => { S.listening = false; say(e.results[0][0].transcript); };
  recog.onerror = () => { S.listening = false; render(); toast('I could not hear that. Try again or type it.'); };
  recog.onend = () => { if (S.listening) { S.listening = false; render(); } };
  S.listening = true; render(); recog.start();
}

/* ---------- navigation and actions ---------- */
async function loadPatients() { S.patients = await call('list_patients', S.q); }
async function goList() { await guard(async () => { await loadPatients(); S.screen = 'list'; S.pid = null; S.pdata = null; S.xray = null; S.xid = null; S.full = false; render(); }); }
async function reloadPatient() { S.pdata = await call('get_patient', S.pid); render(); }
async function openPatient(pid) {
  await guard(async () => {
    S.pid = pid; S.pdata = await call('get_patient', pid); S.screen = 'patient'; S.tab = 'finding'; S.f = 0; S.menu = false; S.writing = false;
    const xs = allXrays(); S.xid = xs.length ? xs[0].id : null; S.xray = S.xid ? await call('get_xray', S.xid) : null; render();
  });
}
async function openXray(id) { await guard(async () => { S.xid = id; S.f = 0; S.menu = false; S.writing = false; S.busy = ''; S.xray = await call('get_xray', id); render(); }); }
function openNewPatient() { guard(async () => { S.modal = { kind: 'patient', nextId: await call('next_patient_id') }; render(); $('#f_name').focus(); }); }
async function uploadFile(file) {
  if (!file || !S.pid) return;
  S.busy = 'Checking what kind of X-ray this is and running the AI models that fit…'; S.tab = 'finding'; render();
  await guard(async () => {
    const data = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(file); });
    const out = await call('add_xray', S.pid, file.name, data);
    S.pdata = await call('get_patient', S.pid);
    S.busy = ''; await openXray(out.id);
    if (S.assist) { S.chat.push({ me: false, t: out.needs_view ? 'I could not tell what kind of X-ray this is. Pick the view on the right and I will run the right checks.' : `This is a ${S.xray.view_label.toLowerCase()}. ${findingList(S.xray).length ? 'Here is what I found.' : 'No foot checks fit it, so nothing was measured.'} ${nextStep()}` }); render(); }
  });
}
async function doAct(act, el) {
  const x = S.xray;
  switch (act) {
    case 'introDone': await call('set_intro_done', true); S.screen = 'login'; render(); return;
    case 'back': return goList();
    case 'logout': await call('logout'); S.doctor = null; S.chat = []; S.assist = false; S.screen = 'login'; S.info = await call('app_info'); render(); return;
    case 'newPatient': return openNewPatient();
    case 'closeModal': S.modal = null; render(); return;
    case 'editPatient': S.modal = { kind: 'patient', edit: true }; render(); return;
    case 'addXray': $('#fileIn').click(); return;
    case 'prevF': { const n = findingList(x).length; S.f = (S.f - 1 + n) % n; break; }
    case 'nextF': { const n = findingList(x).length; S.f = (S.f + 1) % n; break; }
    case 'full': S.full = !S.full; break;
    case 'assist': S.assist = !S.assist; break;
    case 'closeAssist': S.assist = false; if (recog) recog.stop(); S.listening = false; break;
    case 'speak': S.speak = !S.speak; if (!S.speak && window.speechSynthesis) speechSynthesis.cancel(); break;
    case 'mic': toggleMic(); return;
    case 'menu': S.menu = S.menu ? false : true; break;
    case 'changeView': S.menu = 'view'; break;
    case 'unreview': S.menu = false; await guard(async () => { await call('mark_reviewed', x.id, !x.reviewed); x.reviewed = !x.reviewed; await reloadPatient(); }); return;
    case 'deleteXray': S.menu = false; S.modal = { kind: 'confirm' }; break;
    case 'confirmDelete': S.modal = null; await guard(async () => { await call('delete_xray', x.id); S.pdata = await call('get_patient', S.pid); const xs = allXrays(); if (xs.length) await openXray(xs[0].id); else { S.xray = null; S.xid = null; render(); } toast('X-ray deleted'); }); return;
    case 'note': S.writing = true; render(); { const b = $('#noteBox'); if (b) b.focus(); } return;
    case 'cancelNote': S.writing = false; break;
    case 'saveNote': { const t = ($('#noteBox').value || '').trim(); S.writing = false; if (t) await guard(async () => { await call('add_note', S.pid, t); await reloadPatient(); toast("Note saved to today's visit"); }); render(); return; }
    case 'refer': S.modal = { kind: 'refer' }; render(); $('#r_to').focus(); return;
    case 'save': if (!x || x.view === 'unsure') return; await guard(async () => { await call('mark_reviewed', x.id, true); x.reviewed = true; await reloadPatient(); toast('Marked as reviewed and saved'); }); return;
    case 'export': await guard(async () => { const r = await call('export_report', S.pid); toast('Report saved in the patient folder and opened'); return r; }); return;
    case 'openFolder': await guard(() => call('open_patient_folder', S.pid)); return;
    default: return;
  }
  render();
}

/* ---------- render + events ---------- */
function render() {
  const el = $('#screen');
  if (S.screen === 'loading') { el.innerHTML = '<div class="login"><div class="spinner"></div></div>'; return; }
  el.className = 'screen' + (S.screen === 'patient' ? ' patient-screen' : '');
  const chatScroll = $('#msgs') ? $('#msgs').scrollTop : null;
  el.innerHTML = { intro: introScreen, login: loginScreen, list: listScreen, patient: patientScreen }[S.screen]();
  if (S.modal) {
    const m = document.createElement('div');
    m.innerHTML = S.modal.kind === 'patient' ? patientForm(S.modal.edit ? S.pdata.patient : null, S.modal.edit ? S.pdata.history : null)
      : S.modal.kind === 'refer' ? referForm() : confirmForm('Delete this X-ray?', 'confirmDelete');
    el.appendChild(m.firstElementChild);
  }
  const msgs = $('#msgs'); if (msgs) msgs.scrollTop = chatScroll === null ? msgs.scrollHeight : msgs.scrollHeight;
}
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-open],[data-xray],[data-tab],[data-act],[data-chip],[data-say],[data-view]');
  if (!t) { if (S.menu === true && !e.target.closest('.menu')) { S.menu = false; render(); } return; }
  const d = t.dataset;
  if (d.open) return openPatient(d.open);
  if (d.xray) return openXray(+d.xray);
  if (d.tab) { S.tab = d.tab; S.menu = false; return render(); }
  if (d.say) return say(d.say);
  if (d.view) {
    S.busy = 'Running the AI models that fit this X-ray…'; S.menu = false; render();
    return guard(async () => { await call('set_view', S.xid, d.view); S.pdata = await call('get_patient', S.pid); S.busy = ''; await openXray(S.xid); });
  }
  if (d.chip) {
    const k = d.chip;
    if (k === 'bright') S.bright = S.bright >= 90 ? 10 : S.bright + 10;
    if (k === 'contrast') S.contrast = S.contrast >= 90 ? 10 : S.contrast + 10;
    if (k === 'zoom') S.zoom = S.zoom >= 200 ? 100 : S.zoom + 25;
    if (k === 'overlay') S.overlay = (S.overlay + 1) % 3;
    return render();
  }
  doAct(d.act, t);
});
document.addEventListener('submit', async (e) => {
  e.preventDefault();
  const id = e.target.id;
  if (id === 'loginForm' || id === 'setupForm') {
    const err = $('#loginErr');
    try {
      S.doctor = id === 'loginForm' ? await call('login', $('#docId').value, $('#pw').value)
        : await call('create_doctor', $('#docId').value, $('#docName').value, $('#pw').value, $('#clinic').value);
      await goList();
    } catch (x) { err.textContent = x.message; }
  }
  if (id === 'patientForm') {
    const hist = {}; document.querySelectorAll('[data-hist]').forEach(i => { hist[i.dataset.hist] = i.value; });
    const data = { patient_id: $('#f_pid').value, name: $('#f_name').value, age: $('#f_age').value, sex: $('#f_sex').value, phone: $('#f_phone').value, referred_by: $('#f_ref').value, complaint: $('#f_complaint').value, history: hist };
    try {
      if (S.modal.edit) { await call('update_patient', S.pid, data); S.modal = null; await reloadPatient(); toast('Patient details saved'); }
      else { const p = await call('add_patient', data); S.modal = null; await openPatient(p.patient_id); }
    } catch (x) { $('#formErr').textContent = x.message; }
  }
  if (id === 'referForm') {
    const to = $('#r_to').value, reason = $('#r_reason').value;
    S.modal = null; render();
    await guard(async () => { await call('refer_patient', S.pid, to, reason); await reloadPatient(); toast('Referral letter written and opened'); });
  }
  if (id === 'chatForm') { const v = $('#chatIn').value.trim(); if (v) say(v); }
});
document.addEventListener('input', (e) => {
  if (e.target.id === 'listSearch') {
    S.q = e.target.value;
    guard(async () => { await loadPatients(); render(); const i = $('#listSearch'); i.focus(); i.setSelectionRange(S.q.length, S.q.length); });
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.id === 'topSearch') { S.q = e.target.value; goList(); }
  if (e.key === 'Enter' && e.target.matches('tr[data-open]')) e.target.click();
  if (e.key === 'Escape') { if (S.modal) { S.modal = null; render(); } else if (S.menu) { S.menu = false; render(); } }
  if (S.screen === 'patient' && !S.modal && !e.target.matches('input, textarea, select')) {
    if (e.key === 'ArrowRight' && S.xray) doAct('nextF');
    if (e.key === 'ArrowLeft' && S.xray) doAct('prevF');
  }
});
$('#fileIn').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; uploadFile(f); });
document.addEventListener('dragover', (e) => { if (S.screen === 'patient') { e.preventDefault(); const d = $('#drop'); if (d) d.classList.add('over'); } });
document.addEventListener('dragleave', () => { const d = $('#drop'); if (d) d.classList.remove('over'); });
document.addEventListener('drop', (e) => { if (S.screen !== 'patient') return; e.preventDefault(); uploadFile(e.dataTransfer.files[0]); });

(async function start() {
  render();
  try {
    S.info = await call('app_info');
    S.screen = S.info.intro_done ? 'login' : 'intro';
    if (S.info.doctor) { S.doctor = S.info.doctor; await goList(); return; }
  } catch (e) { document.body.innerHTML = `<p style="padding:24px;color:#fff">Remi AI could not start: ${esc(e.message)}</p>`; return; }
  render();
})();
