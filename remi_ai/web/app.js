/* Remi AI window: screens, X-ray viewer, finding panel and the Ask Remi assistant.
   All data and AI work happens in Python (remi_ai/api.py); this file only draws it. */

const ICON = {
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 10V2M5 5l3-3 3 3M3 9v4h10V9"/></svg>',
  search: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>',
  left: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 3 5 8l5 5"/></svg>',
  right: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m6 3 5 5-5 5"/></svg>',
  dots: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="3.5" cy="8" r="1.3"/><circle cx="8" cy="8" r="1.3"/><circle cx="12.5" cy="8" r="1.3"/></svg>',
  expand: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3h5v5M8 17H3v-5M17 3l-6 6M3 17l6-6"/></svg>',
  pen: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M13.5 3.5l3 3L7 16H4v-3z"/><path d="M11.5 5.5l3 3"/></svg>',
  flag: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5 17V3.5"/><path d="M5 4h10l-2.5 3.5L15 11H5"/></svg>',
  eraser: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 16h8M3.5 11.5l7-7 5 5-6.5 6.5H7z"/><path d="M7 8l5 5"/></svg>',
  caret: '<svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m3 4.5 3 3 3-3"/></svg>',
  undo: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M7 5L3.5 8.5 7 12"/><path d="M4 8.5h8a4 4 0 0 1 0 8H9"/></svg>',
  ear: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6 8a4.5 4.5 0 0 1 9 0c0 3-3 3.5-3 6a2.5 2.5 0 0 1-4.6 1.3"/><path d="M8.5 8a2 2 0 0 1 4 0c0 1-1 1.4-1.6 2"/></svg>',
  heat: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 2.5c.5 3-3.5 4.6-3.5 8.6a3.5 3.5 0 0 0 7 0c0-1.6-.8-2.7-1.6-3.6-.2 1.4-.9 2.2-1.8 2.5.6-2.6-.2-5.2-.1-7.5z"/></svg>',
  spark: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="9" cy="9" r="6"/><path d="M13.5 13.5 18 18"/><path d="M9 5.5v7M5.5 9h7" stroke-width="1.3"/></svg>',
  mic: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="5.5" y="1.5" width="5" height="8" rx="2.5"/><path d="M3 7.5a5 5 0 0 0 10 0M8 12.5V15"/></svg>',
  speaker: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2.5 6h2.5l3.5-3v10L5 10H2.5z"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.5a6 6 0 0 1 0 9"/></svg>',
  close: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  info: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6.5"/><path d="M8 7v4.5M8 4.6v.2"/></svg>',
  check: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" width="12" height="12"><path d="m3 8.5 3 3 7-7"/></svg>',
  plus: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 3v10M3 8h10"/></svg>'
};

const S = {
  mark: { on: false, tool: 'pen', color: '#ff5a5f', undo: [], eraser: 'm', emenu: false },
  newDoctor: false,
  screen: 'loading', info: null, doctor: null, patients: [], q: '',
  pid: null, pdata: null, xid: null, xray: null, f: 0, tab: 'finding',
  full: false, assist: false, menu: false, writing: false, busy: '', modal: null,
  bright: 50, contrast: 50, zoom: 100, overlay: 1, heat: false, chat: [], listening: false, speak: false
};
const OVERLAY = ['OFF', 'LINES'];
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
  bunion: 'Bunion angles: within 3° of doctors for 84.4% (HVA) and 95.6% (IMA) of 160 unseen feet.'
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

// Bunion angles have no model probability, so confidence is the chance the grade stays the same given
// the angle error measured in testing (HVA 84.4% and IMA 95.6% within 3°, i.e. about 2.1° and 1.5° spread).
function bandChance(v, sd, cuts) {
  const cdf = z => 1 / (1 + Math.exp(-1.702 * z));
  const lo = [-Infinity, ...cuts].filter(c => c <= v).pop(), hi = [...cuts, Infinity].find(c => c > v);
  return cdf((hi - v) / sd) - cdf((lo - v) / sd);
}
function bunionConfidence(hva, ima) {
  return Math.round(bandChance(hva, 2.1, [15, 20, 40]) * bandChance(ima, 1.5, [9, 11, 16]) * 100);
}
function describe(fd, x) {
  const r = fd.r, prev = prevResult(fd);
  if (fd.key === 'flat_foot') {
    const cat = r.category, measured = cat !== 'Not measured';
    const ang = measured ? r.angle : null, pAng = prev && prev.category !== 'Not measured' ? prev.angle : null;
    const diff = ang != null && pAng != null ? ang - pAng : null;
    return {
      title: cat === 'Not measured' ? 'Arch not measured' : cat,
      sub: `${r.left ? 'Left' : 'Right'} foot · arch height`,
      tags: (measured ? sevTag(r.severity, cat === 'Flat foot' ? 'Low arch' : cat === 'High arch' ? 'High arch' : 'Arch in normal range') : `<span class="tag">${r.tilt ? 'Foot not level' : 'Check the X-ray framing'}</span>`) + '<span class="tag">Side view</span>',
      bar: { label: measured ? `Confidence: ${cat.toLowerCase()}` : `Confidence: ${r.p_flat >= 0.5 ? 'looks flat' : 'does not look flat'}`, pct: Math.round((measured ? (cat === 'Flat foot' ? r.p_flat : 1 - r.p_flat) : Math.max(r.p_flat, 1 - r.p_flat)) * 100), calm: r.severity === 'ok' },
      score: SCORE.flat_foot,
      tiles: [
        { k: 'Heel bone angle', v: ang != null ? ang.toFixed(1) + '°' : '—', h: diff != null ? `${diff > 0 ? '+' : ''}${diff.toFixed(1)}° since ${fmtDate(S.xray.previous.date)}` : measured ? 'Lower means flatter' : r.tilt ? 'Foot not standing level' : 'Could not measure reliably', red: diff != null && diff < 0 },
        { k: 'Normal range', v: '17.7–30°', h: 'Under 17.7° is flat foot' },
        { k: 'Result', v: cat, sm: true, h: !measured ? 'Needs a standing side view' : r.p_flat >= 0.5 === (cat === 'Flat foot') ? 'Both checks agree' : 'Angle and classifier differ' },
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
      bar: { label: `Confidence: ${label === 'spur' ? 'spur seen' : label === 'sever' ? "Sever's" : 'heel clear'}`, pct: Math.round(p[label] * 100), calm: label === 'normal' },
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
    bar: { label: `Confidence: ${grade.toLowerCase()}`, pct: bunionConfidence(r.hva, r.ima), calm: grade === 'No bunion' },
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
let FRAME = [0, 0];  // width and height of the X-ray being drawn, so labels stay inside it
function pill(x, y, text, u, col, flag) {
  if (col === true || col === false) { flag = col; col = ''; }
  const fs = 15 * u, w = (text.length * 0.68 + 1.4) * fs, h = fs * 1.9;
  x = Math.max(8 * u, Math.min(x, FRAME[0] - w - 8 * u));
  y = Math.max(h / 2 + 8 * u, Math.min(y, FRAME[1] - h / 2 - 8 * u));
  return `<rect class="ov-pill${flag ? ' flag' : ''}" x="${x}" y="${y - h / 2}" width="${w}" height="${h}" rx="${7 * u}"/>
    <text class="ov-label ${col || (flag ? 'red' : '')}" x="${x + 0.7 * fs}" y="${y + fs * 0.36}" font-size="${fs}">${esc(text)}</text>`;
}
// A measuring point: a dot, plus a wider invisible grab area the doctor can drag when the finding is selected.
function handle(main, id, cx, cy, u, col) {
  return `<circle class="ov-dot ${col}" cx="${cx}" cy="${cy}" r="${(main ? 4.5 : 3.5) * u}"/>` +
    (main ? `<circle class="ov-grab" data-pt="${id}" cx="${cx}" cy="${cy}" r="${14 * u}"><title>Drag to adjust</title></circle>` : '');
}
// Where the AI looked for this finding (warm = more attention). Bunion angles have no map.
function heatSvg(x, fd) {
  const r = fd.r;
  let out = '';
  if (fd.key === 'flat_foot' && r.heatmap && r.heatmap.length && r.heatmap.some(row => row.some(v => v > 0))) out += `<image href="${heatImage(r.heatmap)}" x="0" y="0" width="${x.width}" height="${x.height}" preserveAspectRatio="none" style="mix-blend-mode:screen"/>`;
  if (fd.key === 'heel_spur' && r.heatmap) { const b = r.crop_box; out += `<image href="${heatImage(r.heatmap)}" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" preserveAspectRatio="none" style="mix-blend-mode:screen"/>`; }
  return out;
}
function hasHeat(fd) { return fd.key !== 'bunion' && heatSvg(S.xray, fd) !== ''; }
function overlaySvg(x, fd, u, main) {
  if (!fd || S.overlay === 0) return '';
  const r = fd.r;
  let out = '';
  if (fd.key === 'flat_foot' && r.points && r.points.L10 && r.category !== 'Not measured') {
    const [x1, y1] = r.points.L10, [x2, y2] = r.points.L11;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    const ex = x1 + dx * 1.25, ey = y1 + dy * 1.25, dir = Math.sign(dx) || 1;
    out += `<line class="ov-dash" x1="${x1 - dir * len * 0.3}" y1="${y1}" x2="${x1 + dir * len * 1.6}" y2="${y1}"/>
      <line class="ov-line c-arch" x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}"/>
      ${handle(main, 'L10', x1, y1, u, 'c-arch')}${handle(main, 'L11', x2, y2, u, 'c-arch')}
      ${pill(Math.min(x1, x2) + Math.abs(dx) * 0.25, Math.max(y1, y2) + 30 * u, `Arch ${r.angle != null ? r.angle.toFixed(1) + '°' : '—'}`, u, 'c-arch', r.severity !== 'ok')}`;
  }
  if (fd.key === 'heel_spur') {
    const [a, b, c, d] = r.box, cx = (a + c) / 2, cy = (b + d) / 2, rx = (c - a) / 2 * 1.05, ry = (d - b) / 2 * 1.05;
    const spur = r.label === 'spur';
    out += `<ellipse class="${spur ? 'ov-red' : 'ov-dash'}" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"/>
      ${pill(cx - rx, b - 26 * u, spur ? 'Heel spur' : r.label === 'sever' ? "Sever's" : 'No spur seen', u, spur)}`;  // above the heel, so it never covers the arch label below
  }
  if (fd.key === 'bunion') {
    // Big toe bone, first and second long bones: each its own colour; HVA and IMA labels match the line they start from.
    const p = r.points, ext = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
    const cols = ['c-toe', 'c-mt1', 'c-mt2'];
    const seg = (i, cls) => { const s = ext(p[i + 1], p[i], 1.15), e = ext(p[i], p[i + 1], 1.15); return `<line class="${cls}" x1="${s[0]}" y1="${s[1]}" x2="${e[0]}" y2="${e[1]}"/>`; };
    out += seg(0, 'ov-line c-toe') + seg(2, 'ov-line c-mt1') + seg(4, 'ov-line c-mt2');
    p.forEach((q, i) => { out += handle(main, i, q[0], q[1], u, cols[i >> 1]); });
    const sideX = r.side === 'left' ? 1 : -1;
    const hx = Math.min(p[0][0], p[1][0], p[2][0]) - 150 * u, hy = p[1][1];
    out += pill(sideX > 0 ? Math.max(p[0][0], p[1][0]) + 30 * u : hx, hy, `HVA ${r.hva.toFixed(1)}°`, u, 'c-toe', r.hva >= 15);
    out += pill(Math.max(p[3][0], p[5][0]) + 30 * u, (p[3][1] + p[5][1]) / 2, `IMA ${r.ima.toFixed(1)}°`, u, 'c-mt2', r.ima >= 9);
  }
  return out;
}
function overlayAll(x, fd, u) {
  if (!fd) return '';
  // Heat map: the selected finding's, or the first other finding that has one (bunion angles have none).
  const list = findingList(x), withHeat = hasHeat(fd) ? fd : list.find(hasHeat);
  const heat = S.heat && withHeat ? heatSvg(x, withHeat) : '';
  if (S.overlay === 0) return heat;
  const others = list.filter(o => !(o.key === fd.key && o.i === fd.i));
  const rest = others.map(o => overlaySvg(x, o, u)).join('');
  return `${heat}<g opacity="0.6">${rest}</g>${overlaySvg(x, fd, u, !x.signed)}`;
}
/* ---------- markup: pen strokes and flag labels the doctor draws on the X-ray ---------- */
const MARK_COLORS = [['#ff5a5f', 'Red'], ['#f5b544', 'Amber'], ['#2ec4b6', 'Teal'], ['#7aa7ff', 'Blue'], ['#f6f5f2', 'White']];
// Erasers, as in OneNote: three sizes that rub out the part of a pen stroke they pass over,
// and a stroke eraser that removes a whole stroke or label on touch. Sizes are screen pixels.
const ERASERS = [['s', 'Small eraser', 5], ['m', 'Medium eraser', 12], ['l', 'Large eraser', 26], ['stroke', 'Stroke eraser', 0]];
const eraserOf = (k) => ERASERS.find(e => e[0] === k) || ERASERS[1];
function eraserIcon(k) {
  if (k === 'stroke') return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 13c3-6 5 2 8-3s4-2 6-4"/><path d="M12 12l5 5M17 12l-5 5"/></svg>';
  const r = { s: 2.5, m: 4.5, l: 7 }[k];
  return `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="${r}"/></svg>`;
}
function eraserMenu() {
  return `<div class="er-menu" role="menu">${ERASERS.map(([k, n]) =>
    `<button role="menuitemradio" aria-checked="${S.mark.eraser === k}" class="${S.mark.eraser === k ? 'on' : ''}" data-act="meraser" data-eraser="${k}">${eraserIcon(k)}<span>${n}</span>${S.mark.eraser === k ? ICON.check : ''}</button>`).join('')}</div>`;
}
function markBar() {
  const t = S.mark.tool, tool = (k, icon, label) => `<button class="${t === k ? 'on' : ''}" data-act="mtool" data-tool="${k}" title="${label}">${icon}<span>${label}</span></button>`;
  const er = eraserOf(S.mark.eraser);
  return `<div class="markbar" role="toolbar" aria-label="Mark up tools">
    ${tool('pen', ICON.pen, 'Pen')}${tool('flag', ICON.flag, 'Label')}
    <span class="er-wrap"><button class="er-main${t === 'erase' ? ' on' : ''}" data-act="mtool" data-tool="erase" title="${er[1]}">${ICON.eraser}<span>${er[1].replace(' eraser', '')}</span></button><button class="er-caret${S.mark.emenu ? ' on' : ''}" data-act="emenu" title="Choose eraser" aria-haspopup="menu" aria-expanded="${S.mark.emenu}">${ICON.caret}</button>${S.mark.emenu ? eraserMenu() : ''}</span>
    <i class="sep"></i>
    ${MARK_COLORS.map(([c, n]) => `<button class="swatch${S.mark.color === c ? ' on' : ''}" data-act="mcolor" data-color="${c}" title="${n}" aria-label="${n}"><b style="background:${c}"></b></button>`).join('')}
    <i class="sep"></i>
    <button data-act="mundo" title="Undo (Ctrl+Z)" ${S.mark.undo.length ? '' : 'disabled'}>${ICON.undo}</button>
    <button data-act="mclear" title="Remove all markings on this X-ray" ${(S.xray.marks || []).length ? '' : 'disabled'}>Clear</button>
    <button class="done" data-act="markup">Done</button>
  </div>`;
}
function flagBox(m, u) {
  const fs = 15 * u, text = m.text || 'Label', w = (text.length * 0.56 + 1.6) * fs, h = fs * 2;
  return { x: m.at[0], y: m.at[1], w, h, fs, text };
}
function marksSvg(x) {
  const u = Math.max(x.width, x.height) / 900;
  return (x.marks || []).map((m, i) => {
    if (m.type === 'pen') {
      const d = m.points.map((p, j) => `${j ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join('');
      return `<path class="mk-pen" data-mark="${i}" d="${d}" stroke="${m.color}"/>`;
    }
    const b = flagBox(m, u), cx = Math.max(b.x, Math.min(m.pin[0], b.x + b.w)), cy = Math.max(b.y, Math.min(m.pin[1], b.y + b.h));
    return `<g class="mk-flag" data-mark="${i}">
      <line class="mk-lead" x1="${m.pin[0]}" y1="${m.pin[1]}" x2="${cx}" y2="${cy}" stroke="${m.color}"/>
      <circle class="mk-pin" data-pin="${i}" cx="${m.pin[0]}" cy="${m.pin[1]}" r="${5 * u}" fill="${m.color}"><title>Drag to move the point</title></circle>
      <g class="mk-label" data-flag="${i}"><title>Drag to move · double-click to edit</title>
        <rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="${6 * u}" stroke="${m.color}"/>
        <text x="${b.x + 0.8 * b.fs}" y="${b.y + b.h / 2 + b.fs * 0.36}" font-size="${b.fs}" fill="${m.text ? '#f6f5f2' : 'rgba(246,245,242,.5)'}">${esc(b.text)}</text></g>
    </g>`;
  }).join('');
}
function redrawMarks() { const g = $('#marks'); if (g && S.xray) g.innerHTML = marksSvg(S.xray); }
function pushUndo() { S.mark.undo.push(JSON.stringify(S.xray.marks || [])); if (S.mark.undo.length > 50) S.mark.undo.shift(); }
// Saves are bundled for a moment while drawing; a pending save is sent at once before another X-ray opens.
let marksTimer = null, marksPending = null;
function flushMarks() {
  clearTimeout(marksTimer); marksTimer = null;
  const p = marksPending; marksPending = null;
  return p ? guard(() => call('save_marks', p.id, p.marks)) : Promise.resolve();
}
function saveMarks() {
  if (S.xray && S.xray.signed) return;
  if (marksPending && marksPending.id !== S.xid) flushMarks();
  marksPending = { id: S.xid, marks: S.xray.marks };
  clearTimeout(marksTimer);
  marksTimer = setTimeout(flushMarks, 250);
}
function svgPoint(svg, e) {
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  return [Math.max(0, Math.min(S.xray.width, p.x)), Math.max(0, Math.min(S.xray.height, p.y))];
}
function editFlag(i, isNew) {
  const m = S.xray.marks[i], el = document.querySelector(`[data-flag="${i}"] rect`); if (!m || !el) return;
  const r = el.getBoundingClientRect(), inp = document.createElement('input');
  inp.className = 'flag-edit'; inp.value = m.text || ''; inp.placeholder = 'Type a label'; inp.maxLength = 200;
  Object.assign(inp.style, { left: r.left + 'px', top: r.top + 'px', minWidth: Math.max(160, r.width) + 'px', height: Math.max(30, r.height) + 'px', borderColor: m.color });
  document.body.appendChild(inp); inp.focus(); inp.select();
  let done = false;
  const finish = (keep) => {
    if (done) return; done = true; inp.remove();
    const t = inp.value.trim();
    if (keep && t) { if (t !== m.text) { if (!isNew) pushUndo(); m.text = t; } }
    else if (isNew && !t) { S.xray.marks.splice(i, 1); }
    redrawMarks(); saveMarks(); render();
  };
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false); e.stopPropagation(); });
  inp.addEventListener('blur', () => finish(true));
}
// Point eraser: cut away the parts of pen strokes within r of p, splitting a stroke in two
// where the eraser crosses its middle. Labels are left alone (the stroke eraser removes them).
function eraseAt(p, r) {
  const marks = S.xray.marks || [], out = [];
  let hit = false;
  const near = (q) => Math.hypot(q[0] - p[0], q[1] - p[1]) <= r;
  for (const m of marks) {
    if (m.type !== 'pen') { out.push(m); continue; }
    const pts = [];  // fill in long segments so the eraser can cut between far-apart points
    m.points.forEach((q, j) => {
      if (j) {
        const a = m.points[j - 1], n = Math.ceil(Math.hypot(q[0] - a[0], q[1] - a[1]) / (r / 2));
        for (let k = 1; k < n; k++) pts.push([a[0] + (q[0] - a[0]) * k / n, a[1] + (q[1] - a[1]) * k / n]);
      }
      pts.push(q);
    });
    if (!pts.some(near)) { out.push(m); continue; }
    hit = true;
    let run = [];
    for (const q of pts) {
      if (near(q)) { if (run.length > 1) out.push({ ...m, points: run }); run = []; }
      else run.push(q);
    }
    if (run.length > 1) out.push({ ...m, points: run });
  }
  if (hit) S.xray.marks = out;
  return hit;
}
// The round outline that follows the pointer while a sized eraser is picked.
document.addEventListener('pointermove', (e) => {
  let c = document.getElementById('ecur');
  const svg = e.target.closest && e.target.closest('svg.xr.marking.tool-erase');
  const r = svg && S.mark.eraser !== 'stroke' ? eraserOf(S.mark.eraser)[2] : 0;
  if (!r) { if (c) c.style.display = 'none'; return; }
  if (!c) { c = document.createElement('div'); c.id = 'ecur'; document.body.appendChild(c); }
  Object.assign(c.style, { display: 'block', width: 2 * r + 'px', height: 2 * r + 'px', left: e.clientX - r + 'px', top: e.clientY - r + 'px' });
});
let MDRAG = null;
document.addEventListener('pointerdown', (e) => {
  if (!S.xray || S.screen !== 'patient') return;
  const svg = e.target.closest && e.target.closest('svg.xr'); if (!svg) return;
  if (S.xray.signed) return;  // a signed-off X-ray's markings stay exactly as they were signed
  const lab = e.target.closest('[data-flag]'), pin = e.target.closest('[data-pin]'), mk = e.target.closest('[data-mark]');
  if (S.mark.on && S.mark.tool === 'erase') {
    e.preventDefault(); e.stopPropagation();
    const flag = mk && S.xray.marks[+mk.dataset.mark] && S.xray.marks[+mk.dataset.mark].type === 'flag';
    if (S.mark.eraser === 'stroke' || flag) {
      if (mk) { pushUndo(); S.xray.marks.splice(+mk.dataset.mark, 1); redrawMarks(); saveMarks(); render(); }
      return;
    }
    const p = svgPoint(svg, e);
    MDRAG = { kind: 'erase', svg, r: eraserOf(S.mark.eraser)[2] / svg.getScreenCTM().a, last: p, changed: false, undo: JSON.stringify(S.xray.marks || []) };
    if (eraseAt(p, MDRAG.r)) { MDRAG.changed = true; redrawMarks(); }
    return;
  }
  if (lab || (pin && S.mark.on)) {
    // Labels can be dragged any time; the pin only while marking up.
    e.preventDefault(); e.stopPropagation();
    const i = +(lab ? lab.dataset.flag : pin.dataset.pin), m = S.xray.marks[i];
    MDRAG = { kind: lab ? 'flag' : 'pin', i, svg, start: svgPoint(svg, e), pin: [...m.pin], at: [...m.at], moved: false, undo: JSON.stringify(S.xray.marks) };
    document.body.classList.add('dragging');
    return;
  }
  if (!S.mark.on) return;
  e.preventDefault(); e.stopPropagation();
  const p = svgPoint(svg, e);
  S.xray.marks = S.xray.marks || [];
  if (S.mark.tool === 'pen') {
    pushUndo();
    S.xray.marks.push({ type: 'pen', color: S.mark.color, points: [p] });
    MDRAG = { kind: 'pen', i: S.xray.marks.length - 1, svg };
  } else if (S.mark.tool === 'flag') {
    pushUndo();
    const u = Math.max(S.xray.width, S.xray.height) / 900;
    S.xray.marks.push({ type: 'flag', color: S.mark.color, text: '', pin: p, at: [p[0] + 28 * u, p[1] - 52 * u] });
    redrawMarks(); editFlag(S.xray.marks.length - 1, true);
  }
}, true);
document.addEventListener('pointermove', (e) => {
  if (!MDRAG) return;
  const p = svgPoint(MDRAG.svg, e);
  if (MDRAG.kind === 'erase') {
    // Rub along the path since the last event so a fast swipe leaves no gaps.
    const d = MDRAG, n = Math.max(1, Math.ceil(Math.hypot(p[0] - d.last[0], p[1] - d.last[1]) / (d.r / 2)));
    let hit = false;
    for (let k = 1; k <= n; k++) hit = eraseAt([d.last[0] + (p[0] - d.last[0]) * k / n, d.last[1] + (p[1] - d.last[1]) * k / n], d.r) || hit;
    d.last = p;
    if (hit) { d.changed = true; redrawMarks(); }
    return;
  }
  const m = S.xray.marks[MDRAG.i]; if (!m) return;
  if (MDRAG.kind === 'pen') {
    const last = m.points[m.points.length - 1], u = Math.max(S.xray.width, S.xray.height) / 900;
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 1.5 * u) m.points.push(p);
  } else {
    const dx = p[0] - MDRAG.start[0], dy = p[1] - MDRAG.start[1];
    if (Math.hypot(dx, dy) > 1) MDRAG.moved = true;
    if (MDRAG.kind === 'flag') { m.pin = [MDRAG.pin[0] + dx, MDRAG.pin[1] + dy]; m.at = [MDRAG.at[0] + dx, MDRAG.at[1] + dy]; }
    else m.pin = [MDRAG.pin[0] + dx, MDRAG.pin[1] + dy];
  }
  redrawMarks();
});
document.addEventListener('pointerup', () => {
  if (!MDRAG) return;
  const d = MDRAG; MDRAG = null; document.body.classList.remove('dragging');
  if (d.kind === 'erase') {
    if (d.changed) { S.mark.undo.push(d.undo); if (S.mark.undo.length > 50) S.mark.undo.shift(); saveMarks(); render(); }
    return;
  }
  if (d.kind === 'pen') {
    const m = S.xray.marks[d.i];
    if (m && m.points.length < 2) m.points.push([m.points[0][0] + 0.5, m.points[0][1] + 0.5]);  // a tap leaves a dot
    saveMarks(); render(); return;
  }
  if (d.moved) { S.mark.undo.push(d.undo); saveMarks(); render(); }
});
document.addEventListener('dblclick', (e) => {
  const lab = e.target.closest && e.target.closest('[data-flag]');
  if (lab && S.xray) { e.preventDefault(); editFlag(+lab.dataset.flag, false); }
});
function xraySvg(x, fd) {
  const u = Math.max(x.width, x.height) / 900;
  FRAME = [x.width, x.height];
  const filt = `filter: brightness(${(0.5 + S.bright / 100).toFixed(2)}) contrast(${(0.5 + S.contrast / 100).toFixed(2)});`;
  return `<svg class="xr${S.mark.on ? ' marking tool-' + S.mark.tool + (S.mark.tool === 'erase' ? ' er-' + S.mark.eraser : '') : ''}" viewBox="0 0 ${x.width} ${x.height}" preserveAspectRatio="xMidYMid meet" style="transform: scale(${S.zoom / 100})" role="img" aria-label="${esc(x.view_label)} X-ray">
    <image href="${x.image}" x="0" y="0" width="${x.width}" height="${x.height}" preserveAspectRatio="none" style="${filt}"/>
    <g id="ov">${overlayAll(x, fd, u)}</g>
    <g id="marks">${marksSvg(x)}</g>
  </svg>`;
}
function lens(image, w, h, region) {
  return `<svg viewBox="${region.map(v => v.toFixed(1)).join(' ')}" preserveAspectRatio="xMidYMid slice"><rect x="-99999" y="-99999" width="199999" height="199999" fill="#000"/><image href="${image}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none"/></svg>`;
}

/* ---------- screens ---------- */
function loginScreen() {
  const first = !S.info.has_doctors || S.newDoctor;
  return `<div class="login"><form class="login-card" id="${first ? 'setupForm' : 'loginForm'}">
    <div class="brand big"><span class="logo"><img src="img/remi_mark.png" alt=""></span><b>Remi</b><span>AI</span></div><div><h2>${!S.info.has_doctors ? 'Create your doctor account' : first ? 'New doctor account' : 'Sign in'}</h2></div>
    ${first ? `<div class="field"><label for="docName">Your name</label><input id="docName" placeholder="Dr. Kavya Rao" required></div>
      <div class="field"><label for="clinic">Clinic</label><input id="clinic" placeholder="Optional"></div>` : ''}
    <div class="field"><label for="docId">Doctor ID</label><input id="docId" autocomplete="username" required placeholder="e.g. DR-2207" value="${first ? '' : esc(S.info.last_doctor_id || '')}"></div>
    <div class="field"><label for="pw">Password</label><input id="pw" type="password" autocomplete="${first ? 'new-password' : 'current-password'}" required placeholder="${first ? 'At least 6 characters' : ''}"></div>
    <div class="err" id="loginErr"></div>
    <button class="btn solid" type="submit">${first ? 'Create account' : 'Sign in'}</button>
    <small>${!S.info.has_doctors ? 'This is the first sign-in on this computer. Patient files are kept here, in your own folder.' : 'Every doctor on this computer sees the same patient list.'}</small>
    ${S.info.has_doctors ? `<button type="button" class="btn" data-act="toggleNewDoctor" style="justify-content:center">${first ? 'Back to sign in' : 'New doctor account'}</button>` : ''}
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
      <div><div class="brand"><span class="logo"><img src="img/remi_mark.png" alt=""></span><b>Remi</b><span>AI</span></div><div class="eyebrow">Signed in as ${esc(S.doctor.name)}</div><h2>Patients</h2></div>
      <div class="list-tools">
        <label class="search">${ICON.search}<input id="listSearch" placeholder="Search name or ID" value="${esc(S.q)}"></label>
        <button class="btn solid" data-act="newPatient">${ICON.plus}New patient</button>
        <button class="btn" data-act="assist">${ICON.spark}Ask Remi</button>
        <button class="avatar" data-act="account" title="${esc(S.doctor.name)} · account">${initials(S.doctor.name)}</button>
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
  const all = list.length > 1 ? `<div class="f-list">${list.map((f, i) => { const d = describe(f, x); return `<button class="${i === S.f ? 'on' : ''}" data-act="pickF" data-i="${i}"><i class="sev ${f.r.category === 'Not measured' ? 'none' : f.r.severity || 'ok'}"></i>${esc(d.title)}</button>`; }).join('')}</div>` : '';
  return `${all}
    <div class="card">
      <div class="f-title">${esc(c.title)}</div><div class="f-sub">${esc(c.sub)}</div>
      <div class="f-tags">${c.tags}${fd.r.edited ? '<span class="tag">Points adjusted by you</span>' : ''}${x.signed ? '<span class="tag">Signed off · locked</span>' : ''}${x.reviewed ? `<span class="tag green">${ICON.check}&nbsp;Reviewed</span>` : ''}</div>
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
const isAssistant = () => S.doctor && S.doctor.role === 'assistant';
function latestVisit() { return S.pdata && S.pdata.visits.length ? S.pdata.visits[0] : null; }
function signButton() {
  const v = latestVisit();
  if (v && v.signed_by) return `<button class="btn solid signed" data-act="openSigned" title="Signed off by ${esc(v.signed_by)}. Click to open the signed report.">${ICON.check}Signed off</button>`;
  if (isAssistant()) return `<button class="btn solid" disabled style="opacity:.45" title="Only a doctor can sign off. You can make the PDF report.">Sign off</button>`;
  return `<button class="btn solid" data-act="signOff" ${v ? '' : 'disabled style="opacity:.5"'} title="Sign off this visit and save a signed PDF report">Sign off</button>`;
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
        ${v.note ? `<p style="white-space:pre-line">${esc(v.note)}</p>` : ''}${!v.xrays.length && !v.note ? '<p>No X-rays or notes</p>' : ''}
        ${v.signed_by ? `<p class="signed-line">${ICON.check} Signed off by ${esc(v.signed_by)} · ${fmtDate(v.signed_at)}</p>` : ''}
        ${v.report ? `<p><a href="#" data-act="openReport" data-file="${esc(v.report)}">${v.signed_by ? 'Signed report' : 'Draft report'} (PDF)</a></p>` : ''}</div></div>`).join('')}
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
      <div class="views" role="tablist">${xs.map(v => `<button class="${v.id === S.xid ? 'on' : ''}" data-xray="${v.id}">${esc(v.view === 'unsure' ? 'Needs view' : v.view_label)} · ${fmtDate(v.date, true)}${v.signed ? '' : `<span class="x" data-act="deleteTab" data-del="${v.id}" role="button" aria-label="Remove this X-ray" title="Remove this X-ray">×</span>`}</button>`).join('')}<button class="add" data-act="addXray">+ Add X-ray</button></div>
    </div>
    ${x ? chips() : ''}
    ${x && S.mark.on ? markBar() : ''}
    <div class="side">
      <div class="toolbar">
        <button class="btn" data-act="export" title="PDF report of the latest visit">${ICON.export}PDF report</button>
        <button class="btn" data-act="save" ${x && x.view !== 'unsure' ? '' : 'disabled style="opacity:.5"'}>${x && x.reviewed ? 'Reviewed' : 'Mark reviewed'}</button>
        ${signButton()}
        <button class="avatar" data-act="account" title="${esc(S.doctor.name)} · account">${initials(S.doctor.name)}</button>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div class="tabs"><button class="${S.tab === 'finding' ? 'on' : ''}" data-tab="finding">${nFind ? `Finding ${String(S.f + 1).padStart(2, '0')}` : 'Findings'}</button><button class="${S.tab === 'record' ? 'on' : ''}" data-tab="record">Record</button></div>
          ${S.tab === 'finding' && x ? `<div class="navs">${nFind > 1 ? `<button class="icon-btn" data-act="prevF" aria-label="Previous finding">${ICON.left}</button><button class="icon-btn" data-act="nextF" aria-label="Next finding">${ICON.right}</button>` : ''}<button class="icon-btn" data-act="menu" aria-label="More">${ICON.dots}</button></div>` : ''}
        </div>
        ${S.menu === true ? (x && x.signed ? `<div class="menu"><button disabled>Signed off · this X-ray is locked</button></div>`
          : `<div class="menu"><button data-act="changeView">Change X-ray type</button><button data-act="unreview">${x && x.reviewed ? 'Mark as not reviewed' : 'Mark as reviewed'}</button><button class="danger" data-act="deleteXray">Delete this X-ray</button></div>`) : ''}
        ${S.tab === 'finding' ? findingPanel() : recordPanel()}
      </div>
    </div>
    <div class="fabs">
      ${x && !x.signed ? `<button class="fab heat mk${S.mark.on ? ' on' : ''}" data-act="markup" aria-label="Mark up" title="Draw and add labels on the X-ray">${ICON.pen}<span>Mark up</span></button>` : ''}
      <button class="fab heat${S.heat ? ' on' : ''}" data-act="heat" aria-label="Heat map" title="Heat map: where the AI looked">${ICON.heat}<span>Heat map</span></button>
      <button class="fab${S.full ? ' on' : ''}" data-act="full" aria-label="Full view" title="Full view">${ICON.expand}</button>
    </div>
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
function signOffForm() {
  const v = latestVisit(), n = v ? v.xrays.length : 0;
  return `<div class="modal-back"><div class="modal"><h3>Sign off ${esc(S.pdata.patient.name)}'s visit?</h3>
    <p>You confirm you have reviewed ${n === 1 ? 'the X-ray' : `all ${n} X-rays`} from ${v ? fmtDate(v.date) : 'this visit'} and Remi's findings. A signed PDF report is saved in the patient's files and a copy in your Downloads folder. Anything added later starts a new visit.</p>
    <div class="actions"><button class="btn" data-act="closeModal">Cancel</button><button class="btn solid" data-act="confirmSignOff">Sign off</button></div></div></div>`;
}
function confirmForm(text, act) {
  return `<div class="modal-back"><div class="modal"><h3>${esc(text)}</h3><p>This removes the X-ray file and its results from the patient's record.</p>
    <div class="actions"><button class="btn" data-act="closeModal">Cancel</button><button class="btn solid" data-act="${act}">Delete</button></div></div></div>`;
}

/* ---------- Ask Remi ---------- */
// Ask Remi floats over every screen: a round Remi button that can be dragged anywhere, and a
// chat window that opens beside it and can be moved by its title bar. Positions are remembered.
const FLOAT = (() => { try { return JSON.parse(localStorage.getItem('remi.float')) || {}; } catch (e) { return {}; } })();
const PANEL_W = 352, BUB = 54;
function panelH() { return Math.min(560, window.innerHeight - 32); }
function clampPos(x, y, w, h) { return [Math.max(8, Math.min(window.innerWidth - w - 8, x)), Math.max(8, Math.min(window.innerHeight - h - 8, y))]; }
function bubblePos() { const b = FLOAT.bubble || [window.innerWidth - BUB - 20, window.innerHeight - BUB - 92]; return clampPos(b[0], b[1], BUB, BUB); }
function panelPos() {
  if (FLOAT.panel) return clampPos(FLOAT.panel[0], FLOAT.panel[1], PANEL_W, panelH());
  const [bx, by] = bubblePos();  // first time: open beside the button, on whichever side has room
  const x = bx > window.innerWidth / 2 ? bx - PANEL_W - 12 : bx + BUB + 12;
  return clampPos(x, by + BUB - panelH(), PANEL_W, panelH());
}
function saveFloat() { try { localStorage.setItem('remi.float', JSON.stringify(FLOAT)); } catch (e) { /* positions just reset next time */ } }
function floatingRemi() {
  if (!S.doctor || S.screen === 'login') return '';
  const [bx, by] = bubblePos();
  return `<button class="remi-bubble${S.assist ? ' on' : ''}${S.listening || V.handsfree ? ' live' : ''}" id="remiBubble" style="left:${bx}px;top:${by}px" aria-label="Ask Remi" title="Ask Remi · drag to move"><img src="img/remi_mark.png" alt=""></button>
    ${S.assist ? assistPanel() : ''}`;
}
let FDRAG = null;
document.addEventListener('pointerdown', (e) => {
  const bub = e.target.closest && e.target.closest('#remiBubble');
  const head = !bub && e.target.closest && e.target.closest('.assist-head');
  if (!bub && !(head && !e.target.closest('button'))) return;
  const el = bub || head.closest('.assist'), r = el.getBoundingClientRect();
  FDRAG = { el, kind: bub ? 'bubble' : 'panel', dx: e.clientX - r.left, dy: e.clientY - r.top, sx: e.clientX, sy: e.clientY, moved: false };
  e.preventDefault();
});
document.addEventListener('pointermove', (e) => {
  if (!FDRAG) return;
  if (Math.hypot(e.clientX - FDRAG.sx, e.clientY - FDRAG.sy) > 4) FDRAG.moved = true;
  if (!FDRAG.moved) return;
  const r = FDRAG.el.getBoundingClientRect(), [x, y] = clampPos(e.clientX - FDRAG.dx, e.clientY - FDRAG.dy, r.width, r.height);
  FDRAG.el.style.left = x + 'px'; FDRAG.el.style.top = y + 'px';
  FLOAT[FDRAG.kind] = [x, y];
  document.body.classList.add('dragging');
});
document.addEventListener('pointerup', () => {
  if (!FDRAG) return;
  const d = FDRAG; FDRAG = null; document.body.classList.remove('dragging');
  if (d.moved) saveFloat();
  else if (d.kind === 'bubble') doAct('assist');
});
window.addEventListener('resize', () => { if (S.doctor && S.screen !== 'login') render(); });
function assistPanel() {
  if (!S.chat.length) S.chat.push({ me: false, t: `Hi ${S.doctor.name.split(' ').slice(0, 2).join(' ')}. I can guide you through Remi step by step. ${nextStep()}` });
  const sugg = S.screen === 'patient'
    ? ['What do I do next?', 'Explain this finding', 'Why was a check skipped?', 'Show the record']
    : ['What do I do next?', 'Add a new patient', 'How accurate are the models?'];
  const [px, py] = panelPos();
  return `<div class="assist" role="dialog" aria-label="Ask Remi" style="left:${px}px;top:${py}px;height:${panelH()}px">
    <div class="assist-head"><span class="logo sm"><img src="img/remi_mark.png" alt=""></span><div style="flex:1"><b>Ask Remi</b><span>${voiceLine()}</span></div>
      <div style="display:flex;gap:6px"><button class="icon-btn hf${V.handsfree ? ' on' : ' dim'}" data-act="handsfree" aria-label="Hands-free" title="Hands-free: keep listening for &quot;Remi, …&quot;">${ICON.ear}</button><button class="icon-btn${S.speak ? '' : ' dim'}" data-act="speak" aria-label="Read all answers aloud" title="Read all answers aloud (voice questions are always answered aloud)">${ICON.speaker}</button><button class="icon-btn" data-act="closeAssist" aria-label="Close">${ICON.close}</button></div></div>
    <div class="msgs" id="msgs">${S.chat.map(m => `<div class="msg ${m.me ? 'me' : 'bot'}">${esc(m.t)}</div>`).join('')}</div>
    <div class="sugg">${sugg.map(s => `<button data-say="${s}">${s}</button>`).join('')}</div>
    <form class="composer" id="chatForm"><button type="button" class="mic${S.listening ? ' live' : ''}" data-act="mic" aria-label="Speak" title="Speak a command (F2)" style="--lvl:${V.level}">${ICON.mic}</button>
      <input id="chatIn" placeholder="${S.listening ? 'Listening… speak now' : 'Type, or press F2 and speak'}" autocomplete="off"><button class="btn solid" type="submit">Send</button></form>
  </div>`;
}
function nextStep() {
  if (S.screen === 'list') return S.patients.length ? 'Open a patient from the list, or say "new patient" to add one.' : 'Start by adding your first patient: say "new patient" or press New patient.';
  if (S.screen === 'patient') {
    const x = S.xray;
    if (!x) return 'Add an X-ray for this patient: press Add X-ray or drop the file on the left.';
    if (x.view === 'unsure') return 'Tell me what kind of X-ray this is using the three buttons on the right, so I run the right checks.';
    if (!findingList(x).length) return 'No checks ran on this X-ray. Add a side or top-down foot X-ray, or change the X-ray type from the menu.';
    if (!x.reviewed) return 'Look at the lines on the X-ray and the numbers on the right. If you agree, press Mark reviewed. When the visit is done, a doctor presses Sign off to save the signed PDF report.';
    return latestVisit() && latestVisit().signed_by ? 'This visit is signed off and the PDF report is in the patient\'s files. Anything new you add starts a new visit.' : isAssistant() ? 'This X-ray is reviewed. Press PDF report to prepare the report; the doctor signs it off.' : 'This X-ray is reviewed. Add a note or referral if needed, then press Sign off to save the signed PDF report.';
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
/* Adding patients by voice or text: "add a patient named Amir, 34, male, heel pain for 2 months".
   Remi creates the record as soon as it has a name, then asks for anything still missing. */
const HIST_KEYS = [[/diabet|sugar/i, 'Diabetes'], [/\bbp\b|blood pressure|hypertension/i, 'Blood pressure'],
  [/medicin|medication|tablet|drugs?\b/i, 'Medicines'], [/surgery|operat/i, 'Past foot surgery'], [/allerg/i, 'Allergies']];
const capFirst = t => t ? t.charAt(0).toUpperCase() + t.slice(1) : t;
const titleCase = t => t.split(/\s+/).map(w => w ? w.charAt(0).toUpperCase() + w.slice(1) : w).join(' ');
const STOP = String.raw`(?=,|\.|;|\s+aged?\b|\s+who\b|\s+with\b|\s+(?:fe)?male\b|\s+phone\b|\s+mobile\b|\s+complain|\s+\d|\s+and\b|\s+age\b|\s+is\s+\d|\s*$)`;
function parseDetails(text) {
  const d = { history: {} }, t = ' ' + text.replace(/\s+/g, ' ').trim() + ' ';
  let m;
  if ((m = t.match(new RegExp(String.raw`\b(?:named|called|name is|name)\s+([a-z][a-z.'-]*(?:\s+(?!and\b|aged?\b|who\b|with\b|male\b|female\b|phone\b|age\b|is\b)[a-z][a-z.'-]*){0,3}?)` + STOP, 'i')))) d.name = titleCase(m[1].trim());
  if ((m = t.match(/\b(?:aged?|age is|age:)\s*(\d{1,3})\b/i) || t.match(/\b(\d{1,3})\s*(?:years?|yrs?|y\/o|yo)\b/i) || t.match(/(?:^|,)\s*(\d{1,3})\s*(?:,|$)/))) d.age = m[1];
  if (/\b(female|woman|girl|lady)\b/i.test(t)) d.sex = 'Female'; else if (/\b(male|man|boy|gentleman)\b/i.test(t)) d.sex = 'Male';
  if ((m = t.match(/\b(?:phone|mobile|number|contact)(?:\s+(?:number|no\.?))?(?:\s+is)?\s*[:-]?\s*(\+?\d[\d\s-]{5,}\d)/i))) d.phone = m[1].replace(/\s+/g, ' ').trim();
  if ((m = t.match(/\breferred by\s+([^,.;]+)/i))) d.referred_by = m[1].trim();
  if ((m = t.match(/\b(?:complaint(?: is)?|complains? of|complaining of|came (?:in )?(?:with|for)|problem is|suffering from|presenting with|has)\s*[:-]?\s*([^,.;]+)/i)) && !HIST_KEYS.some(([re]) => re.test(m[1]))) d.complaint = capFirst(m[1].trim());
  else if ((m = t.match(/\b((?:heel|foot|ankle|arch|toe|big toe|knee)\s+(?:pain|swelling|ache|problem|injury)[^,.;]*)/i))) d.complaint = capFirst(m[1].trim());
  text.split(/[,.;]|\band\b/i).forEach(seg => {
    for (const [re, item] of HIST_KEYS) {
      if (!re.test(seg)) continue;
      if (/\b(no|none|nil|not|negative)\b/i.test(seg)) { d.history[item] = 'No'; continue; }
      const rest = seg.replace(new RegExp('(?:' + re.source + String.raw`)\w*`, 'i'), '').replace(/^[\s\w]*?\b(is|are|:|-)\s*/i, '').replace(/^\s*(yes|has|have|on|takes?)\s*/i, '').trim();
      d.history[item] = rest ? capFirst(rest.replace(/^(mellitus|pressure)\s*/i, '')) || 'Yes' : 'Yes';
    }
  });
  return d;
}
const INTAKE = [
  ['name', d => "What is the patient's full name?"],
  ['age', d => `How old is ${d.first}?`],
  ['sex', d => 'Female, male or other?'],
  ['phone', d => 'Phone number? Say skip if you don\'t have it.'],
  ['complaint', d => `What did ${d.first} come in with? For example, heel pain for three months.`],
  ['history', d => 'Any medical history: diabetes, blood pressure, medicines, past foot surgery or allergies? Say none if not.'],
];
function intakeNext() {
  const it = S.intake, p = it.pid && S.pdata && S.pdata.patient;
  const have = { name: p ? p.name : it.data.name, age: p ? p.age : it.data.age, sex: p ? p.sex : it.data.sex, phone: p ? p.phone : it.data.phone,
    complaint: p ? p.complaint : it.data.complaint, history: p ? Object.keys(S.pdata.history).length : 0 };
  const step = INTAKE.find(([k]) => !have[k] && !it.skipped.includes(k));
  if (!step) { const n = (p && p.name) || it.data.name; S.intake = null; return `All done. ${n}'s record is saved. Add their first X-ray when you're ready: say "add X-ray".`; }
  it.step = step[0];
  return step[1]({ first: ((p && p.name) || it.data.name || 'the patient').split(' ')[0] });
}
async function savePatientDetails(d) {
  const fields = {};
  ['name', 'age', 'sex', 'phone', 'referred_by', 'complaint'].forEach(k => { if (d[k]) fields[k] = d[k]; });
  if (Object.keys(d.history || {}).length) fields.history = { ...S.pdata.history, ...d.history };
  if (!Object.keys(fields).length) return false;
  await call('update_patient', S.pid, fields); S.pdata = await call('get_patient', S.pid);
  return Object.keys(fields).map(k => k === 'history' ? Object.keys(d.history).join(', ').toLowerCase() : k.replace('_', ' ')).join(', ');
}
async function createPatient(d) {
  const id = await call('next_patient_id');
  const p = await call('add_patient', { patient_id: id, name: d.name, age: d.age || '', sex: d.sex || '', phone: d.phone || '', referred_by: d.referred_by || '', complaint: d.complaint || '', history: d.history || {} });
  await openPatient(p.patient_id); S.assist = true;
  return p;
}
async function intakeAnswer(q) {
  const it = S.intake, s = q.toLowerCase().trim();
  if (/^(cancel|stop|quit|never ?mind)\b/.test(s)) { S.intake = null; return it.pid ? 'Stopped. Everything you gave me so far is saved.' : 'Cancelled. No patient was added.'; }
  if (/^(skip|don'?t know|no idea|not now|later|pass)\b/.test(s) && it.step !== 'history') { it.skipped.push(it.step); return intakeNext(); }
  const d = parseDetails(q);
  if (it.step === 'name' && !d.name) d.name = titleCase(q.replace(/^(it'?s|the name is|name is|his name is|her name is)\s+/i, '').replace(/[.,!]+$/, '').trim());
  if (it.step === 'age' && !d.age) { const m = q.match(/\d{1,3}/); if (m) d.age = m[0]; else return 'Please say the age as a number, or say skip.'; }
  if (it.step === 'sex' && !d.sex) { if (/other|prefer not/i.test(q)) d.sex = 'Other'; else return 'Female, male or other? You can also say skip.'; }
  if (it.step === 'phone' && !d.phone) { const m = q.match(/\+?\d[\d\s-]{5,}\d/); if (m) d.phone = m[0]; else return 'Please say the phone number digits, or say skip.'; }
  if (it.step === 'complaint' && !d.complaint) d.complaint = capFirst(q.replace(/[.]+$/, '').trim());
  if (it.step === 'history' && !Object.keys(d.history).length) {
    if (/^(none|no|nothing|nil|no history)\b/i.test(s)) { it.skipped.push('history'); return intakeNext(); }
    d.history = { 'Other history': capFirst(q.trim()) };
  }
  if (!it.pid) {
    if (!d.name) return "I didn't catch a name. What is the patient's full name?";
    const p = await createPatient(d); it.pid = p.patient_id;
    const n = intakeNext();
    return `Added ${p.name} as patient ${p.patient_id}. ` + n;
  }
  await savePatientDetails(d);
  return 'Saved. ' + intakeNext();
}
async function patientCommand(q, s) {
  // "add a patient named Amir ..." creates the record straight away; "add a patient" starts a short interview.
  if (/\b(form)\b/.test(s)) { openNewPatient(); return 'Opened the new patient form.'; }
  const d = parseDetails(q);
  if (!d.name) { S.intake = { step: 'name', data: {}, skipped: [] }; return "Let's add a new patient. " + intakeNext(); }
  const p = await createPatient(d);
  S.intake = { pid: p.patient_id, data: {}, skipped: [] };
  const got = ['age', 'sex', 'phone', 'complaint'].filter(k => d[k]).concat(Object.keys(d.history).length ? ['history'] : []);
  return `Added ${p.name} as patient ${p.patient_id}${got.length ? ` with ${got.join(', ')}` : ''}. ` + intakeNext();
}
function answer(q) {
  const s = q.toLowerCase().trim();
  const onPatient = S.screen === 'patient';
  let m;
  if (/^(stop listening|hands[ -]?free off|stop hands[ -]?free|go to sleep)\b/.test(s)) { stopListening(); return 'Okay, I stopped listening.'; }
  if (S.intake) return intakeAnswer(q);
  if (/(change|switch|next|different|another|new) (your )?voice/.test(s)) {
    const vs = voiceList(); if (vs.length < 2) return 'This computer has only one English voice. You can add more in Windows Settings, under Time and language, Speech.';
    const cur = pickVoice(), next = vs[(vs.indexOf(cur) + 1) % vs.length];
    VOICE.name = next.name; saveVoice();
    const reply = `This is my new voice: ${voiceLabel(next).replace(/ · /g, ', ')}. Say change voice again to hear another, or pick one on your account page.`;
    setTimeout(() => speakText(reply), 0);
    return reply;
  }
  if (/\b(new|add|create|register|make)\b.*\bpatient\b/.test(s)) return patientCommand(q, s);
  if (onPatient && (/\b(add|update|change|set|fill)\b.*\b(details?|info|history|age|phone|complaint|sex|gender)\b/.test(s) || /^(age|phone|mobile|complaint|diabetes|blood pressure|bp|medicines?|allerg|sex|gender)\b/.test(s))) {
    return (async () => {
      const got = await savePatientDetails(parseDetails(q));
      if (got) return `Updated ${S.pdata.patient.name}'s ${got}.`;
      S.intake = { pid: S.pid, data: {}, skipped: [] };
      return intakeNext();
    })();
  }
  if ((m = s.match(/^(open|show|find)\s+(patient\s+)?(.+)$/)) && !/(record|history|side|top|finding|folder|report)/.test(s)) {
    const term = m[3].trim(); const p = S.patients.find(p => p.name.toLowerCase().includes(term) || p.patient_id.toLowerCase() === term);
    if (p) { openPatient(p.patient_id); return `Opening ${p.name}.`; }
    return `I couldn't find a patient called "${m[3]}". Check the spelling or the patient ID.`;
  }
  if (/(patients|patient list|go back|^back)/.test(s)) { goList(); return 'Back to the patient list.'; }
  if (/accura|how good|trust/.test(s)) return 'In testing: flat foot 93.8% correct on 211 unseen X-rays; heel spur 97.5% on 595; bunion angles within 3° of doctors for 84.4% (HVA) and 95.6% (IMA) of 160 feet; the X-ray check picks the right view for 98.7% of 1,738. Always check the result yourself.';
  if (/explain|what does|mean|angle|hva|ima/.test(s)) return onPatient ? explainFinding() : 'Open a patient and an X-ray, then ask me to explain a finding.';
  if (/finding/.test(s) && /next/.test(s) && onPatient) { const n = findingList(S.xray).length; if (n < 2) return 'This X-ray has only one finding.'; S.f = (S.f + 1) % n; S.tab = 'finding'; return 'Next finding.'; }
  if (/(what|next|help|guide|how do i|start)/.test(s)) return nextStep();
  if (!onPatient) return 'Open a patient first. Then I can explain findings, switch X-rays, add notes and write referrals.';
  if (/(add|upload|new).*x-?ray|upload/.test(s)) { $('#fileIn').click(); return 'Choose the X-ray file. I will check what kind of X-ray it is and run only the models that fit.'; }
  if (/skip|not run|why.*(no|not)/.test(s)) { const x = S.xray; const sk = x && x.results.skipped ? Object.entries(x.results.skipped) : []; return sk.length ? sk.map(([k, v]) => `${S.info.checks[k].name}: ${v.replace(/^Not run: /, 'not run because it ')}`).join('. ') + '. Each model only runs on the X-ray view it was trained on.' : 'Nothing was skipped on this X-ray.'; }
  if (/side/.test(s) || /top/.test(s)) { const want = /side/.test(s) ? 'side' : 'top_down'; const x = allXrays().find(v => v.view === want); if (x) { openXray(x.id); return `Opened the ${want === 'side' ? 'side' : 'top-down'} view.`; } return `This patient has no ${want === 'side' ? 'side' : 'top-down'} X-ray yet.`; }
  if (/record|history/.test(s)) { S.tab = 'record'; return 'The record is open on the right: complaint, medical history and every visit.'; }
  if (/finding/.test(s)) { S.tab = 'finding'; return 'Showing the findings.'; }
  if ((m = q.match(/^\s*note[:,]?\s+(.+)/i))) { guard(async () => { await call('add_note', S.pid, m[1]); await reloadPatient(); }); return 'Saved that note to today\'s visit.'; }
  if (/sign(ed)?[ -]?off/.test(s)) { if (isAssistant()) return 'Only a doctor can sign off. You can prepare the PDF report: say "make the report".'; doAct('signOff'); return 'Check the summary and press Sign off to confirm.'; }
  if (/save|review/.test(s)) { doAct('save'); return 'Marked this X-ray as reviewed.'; }
  if (/export|report|pdf/.test(s)) { doAct('export'); return 'Making the PDF report. It is saved in the patient\'s files and your Downloads folder, and opens when ready.'; }
  if (/refer/.test(s)) { doAct('refer'); return 'Opened the referral form.'; }
  if (/zoom in/.test(s)) { S.zoom = Math.min(200, S.zoom + 25); return 'Zoomed in.'; }
  if (/zoom out/.test(s)) { S.zoom = Math.max(50, S.zoom - 25); return 'Zoomed out.'; }
  if (/bright/.test(s)) { S.bright = Math.min(100, S.bright + 10); return 'Brighter.'; }
  if (/dark/.test(s)) { S.bright = Math.max(0, S.bright - 10); return 'Darker.'; }
  if (/(heat|map|focus).*\b(off|hide)\b|\b(hide|turn off)\b.*(heat|map)/.test(s)) { S.heat = false; return 'Heat map off.'; }
  if (/heat|map|focus/.test(s)) { S.heat = true; return 'Showing where the AI looked (the warm areas).'; }
  if (/mark ?up|draw|annotate|eras|rub out/.test(s) && S.xray && S.xray.signed) return 'This X-ray is signed off, so its markings are locked. Add a new X-ray to start a new visit.';
  if (/eraser|erase|rub out/.test(s) && S.xray) {
    const k = /stroke|whole/.test(s) ? 'stroke' : /small|little|tiny/.test(s) ? 's' : /large|big/.test(s) ? 'l' : /medium/.test(s) ? 'm' : S.mark.eraser;
    S.mark.on = true; S.mark.tool = 'erase'; S.mark.eraser = k;
    return `${eraserOf(k)[1]} on. ${k === 'stroke' ? 'Touch a line or label to remove it.' : 'Rub over a line to erase that part.'}`;
  }
  if (/mark ?up|draw|annotate/.test(s)) { S.mark.on = true; return 'Mark up is on. Pick pen or label at the top of the X-ray.'; }
  if (/next (x-?ray|image)/.test(s)) { const xs = allXrays(), i = xs.findIndex(v => v.id === S.xid); if (xs.length > 1) { openXray(xs[(i + 1) % xs.length].id); return 'Next X-ray.'; } return 'This patient has only one X-ray.'; }
  if (/hide.*line|lines off/.test(s)) { S.overlay = 0; return 'Lines hidden.'; }
  if (/line/.test(s)) { S.overlay = 1; return 'Lines shown.'; }
  if (/full|hide panel/.test(s)) { S.full = true; return 'Full view. Press the expand button to bring the panel back.'; }
  return 'I can: add a patient (say "add a patient named ..."), fill in details ("age 34, phone ..."), explain this finding, say why a check was skipped, open the side or top view, add a note (say "note: ..."), save, export a report, refer the patient, zoom, or show where the AI looked.';
}
async function say(text, byVoice) {
  S.chat.push({ me: true, t: text }); render();
  let a;
  try { a = await answer(text); } catch (e) { a = e.message || 'Something went wrong.'; }
  S.chat.push({ me: false, t: a });
  // Voice questions get spoken answers. During a voice interview Remi listens again after asking.
  const after = () => {
    if (!byVoice) return;
    if (V.handsfree) V.followUp = Date.now() + 8000;
    else if (S.intake && !S.listening) startListening('once');
  };
  if (byVoice || S.speak) speakText(a, after); else after();
  render();
  const i = $('#chatIn'); if (i && !byVoice) i.focus();
}

/* ---------- voice: offline speech recognition in Python (remi_ai/voice.py) + Windows voices ---------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;  // only used in the browser dev mode
let recog = null;
const V = { poll: null, handsfree: false, level: 0, partial: '', quietUntil: 0, followUp: 0, status: null };
// What the recogniser may hear for "Remi" with an Indian English model.
const WAKE = /^(?:(?:hey|hi|ok|okay|hello)\s+)?(?:remi|remy|rimi|ramy|reemi|remmy|romy|rummy|army|ready|ruby)\b[\s,]*/i;
function voiceLine() {
  const st = V.status;
  if (V.handsfree) return 'Hands-free on · say "Remi, …"';
  if (S.listening) return V.partial ? `Hearing: ${esc(V.partial)}` : 'Listening… speak now';
  if (st && st.downloading != null) return `Setting up offline voice… ${st.downloading}%`;
  return 'Voice or text · press F2 to speak';
}
// Remi's spoken voice: any English voice installed in Windows, chosen on the account page or by
// saying "change voice". The choice, speed and pitch are remembered on this computer.
const VOICE = (() => { try { return JSON.parse(localStorage.getItem('remi.voice')) || {}; } catch (e) { return {}; } })();
function saveVoice() { try { localStorage.setItem('remi.voice', JSON.stringify(VOICE)); } catch (e) { /* falls back to the default voice */ } }
const isNatural = (v) => /natural|online|neural/i.test(v.name);
function voiceList() {
  const vs = window.speechSynthesis ? speechSynthesis.getVoices().filter(v => /^en/i.test(v.lang)) : [];
  const rank = (v) => (isNatural(v) ? 0 : 4) + (/en-IN/i.test(v.lang) ? 0 : /en-GB/i.test(v.lang) ? 1 : 2);
  return vs.sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
}
function pickVoice() {
  const vs = voiceList();
  return (VOICE.name && vs.find(v => v.name === VOICE.name)) || vs[0] || null;
}
function voiceLabel(v) {
  const accent = { 'en-IN': 'Indian English', 'en-GB': 'British English', 'en-US': 'American English', 'en-AU': 'Australian English' }[v.lang.replace('_', '-')] || v.lang;
  return `${v.name.replace(/^Microsoft\s+/, '').replace(/\s*-\s*English.*$/, '').replace(/\s*\(.*?\)\s*/g, ' ').trim()} · ${accent}${isNatural(v) ? ' · natural' : ''}`;
}
const VOICE_SAMPLE = 'Hi, I am Remi. I will guide you through each X-ray, step by step.';
function speakText(text, then) {
  if (!window.speechSynthesis) { if (then) then(); return; }
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(/°/g, ' degrees'));
  const v = pickVoice(); if (v) { u.voice = v; u.lang = v.lang; }
  u.rate = VOICE.rate || 1; u.pitch = VOICE.pitch || 1;
  V.quietUntil = Date.now() + 60000;  // don't act on Remi's own voice coming back through the mic
  u.onend = u.onerror = () => { V.quietUntil = Date.now() + 500; if (then) then(); };
  speechSynthesis.speak(u);
}
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => { if (S.screen === 'account') render(); };
function updateVoiceUi() {
  const mic = $('.composer .mic'); if (mic) { mic.style.setProperty('--lvl', V.level); mic.classList.toggle('live', S.listening || V.handsfree); }
  const line = $('.assist-head span:not(.logo)'); if (line) line.textContent = voiceLine().replace(/&quot;/g, '"');
  const inp = $('#chatIn'); if (inp && S.listening) inp.placeholder = V.partial ? V.partial + '…' : 'Listening… speak now';
  const pill = $('#hfPill'); if (pill) pill.style.setProperty('--lvl', V.level);
}
async function startListening(mode) {
  if (window.speechSynthesis && speechSynthesis.speaking && mode === 'once') speechSynthesis.cancel();
  try {
    await call('voice_start', mode);
  } catch (e) {
    if (mode === 'once' && SR && window.REMI_DEV) return webSpeechOnce();
    V.status = await call('voice_status').catch(() => V.status);
    toast(e.message); render(); return;
  }
  if (mode === 'handsfree') V.handsfree = true; else S.listening = true;
  V.partial = '';
  if (!V.poll) V.poll = setInterval(pollVoice, 150);
  render();
}
async function stopListening() {
  V.handsfree = false; S.listening = false; V.partial = '';
  await call('voice_stop').catch(() => {});
  render();
}
async function pollVoice() {
  let r;
  try { r = await call('voice_poll'); } catch (e) { return; }
  V.level = r.level || 0; V.partial = r.partial || '';
  for (const t of r.finals) heard(t);
  if (!r.listening) {
    clearInterval(V.poll); V.poll = null; V.level = 0;
    const was = S.listening || V.handsfree; S.listening = false; V.handsfree = false; V.partial = '';
    if (was) render();
    return;
  }
  updateVoiceUi();
}
function heard(text) {
  if (Date.now() < V.quietUntil) return;
  if (!V.handsfree) { S.listening = false; S.assist = true; return say(text, true); }
  const m = text.match(WAKE);
  if (m) {
    const cmd = text.slice(m[0].length).trim();
    S.assist = true;
    if (!cmd) { V.followUp = Date.now() + 8000; render(); return speakText('Yes?'); }
    return say(cmd, true);
  }
  // Right after Remi asks something (or during an interview) the answer needs no "Remi".
  if (Date.now() < V.followUp || S.intake) { S.assist = true; return say(text, true); }
}
function webSpeechOnce() {
  recog = new SR(); recog.lang = 'en-IN'; recog.interimResults = false;
  recog.onresult = (e) => { S.listening = false; say(e.results[0][0].transcript, true); };
  recog.onerror = () => { S.listening = false; render(); toast('I could not hear that. Try again or type it.'); };
  recog.onend = () => { if (S.listening) { S.listening = false; render(); } };
  S.listening = true; render(); recog.start();
}
function toggleMic() {
  if (S.listening) { if (recog) recog.stop(); return stopListening(); }
  if (V.handsfree) { V.followUp = Date.now() + 8000; toast('Listening. Go ahead.'); return; }
  S.assist = true; startListening('once');
}
function handsfreePill() {
  if (!V.handsfree || !S.doctor) return '';
  return `<button class="hf-pill" id="hfPill" data-act="handsfree" title="Turn hands-free off" style="--lvl:${V.level}"><i></i>Hands-free · say “Remi, …”</button>`;
}

/* ---------- navigation and actions ---------- */
async function loadPatients() { S.patients = await call('list_patients', S.q); }
async function goList() { await flushMarks(); await guard(async () => { await loadPatients(); S.screen = 'list'; S.pid = null; S.pdata = null; S.xray = null; S.xid = null; S.full = false; render(); }); }
async function reloadPatient() { S.pdata = await call('get_patient', S.pid); render(); }
async function openPatient(pid) {
  await guard(async () => {
    S.pid = pid; S.pdata = await call('get_patient', pid); S.screen = 'patient'; S.tab = 'finding'; S.f = 0; S.menu = false; S.writing = false;
    const xs = allXrays(); S.xid = xs.length ? xs[0].id : null; S.xray = S.xid ? await call('get_xray', S.xid) : null; render();
  });
}
async function openXray(id) { await flushMarks(); await guard(async () => { S.mark.undo = []; S.mark.on = false; S.xid = id; S.f = 0; S.menu = false; S.writing = false; S.busy = ''; S.xray = await call('get_xray', id); render(); }); }
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
    case 'back': return goList();
    case 'toggleNewDoctor': S.newDoctor = !S.newDoctor; render(); return;
    case 'logout': if (S.listening || V.handsfree) await stopListening(); await call('logout'); S.doctor = null; S.chat = []; S.assist = false; S.screen = 'login'; S.info = await call('app_info'); render(); return;
    case 'newPatient': return openNewPatient();
    case 'closeModal': S.modal = null; render(); return;
    case 'editPatient': S.modal = { kind: 'patient', edit: true }; render(); return;
    case 'addXray': $('#fileIn').click(); return;
    case 'prevF': { const n = findingList(x).length; S.f = (S.f - 1 + n) % n; break; }
    case 'pickF': S.f = +el.dataset.i; break;
    case 'nextF': { const n = findingList(x).length; S.f = (S.f + 1) % n; break; }
    case 'heat': S.heat = !S.heat; if (S.heat && x && !findingList(x).some(hasHeat)) toast('No heat map for this X-ray. Bunion angles are measured from points, not a map.'); break;
    case 'markup': S.mark.on = !S.mark.on; S.mark.undo = []; if (S.mark.on) { S.overlay = S.overlay || 1; } break;
    case 'mtool': S.mark.emenu = false; S.mark.tool = el.dataset.tool; break;
    case 'emenu': S.mark.emenu = !S.mark.emenu; break;
    case 'meraser': S.mark.eraser = el.dataset.eraser; S.mark.tool = 'erase'; S.mark.emenu = false; break;
    case 'mcolor': S.mark.emenu = false; S.mark.color = el.dataset.color; if (S.mark.tool === 'erase') S.mark.tool = 'pen'; break;
    case 'mundo': if (S.mark.undo.length) { S.xray.marks = JSON.parse(S.mark.undo.pop()); saveMarks(); } break;
    case 'mclear': if ((S.xray.marks || []).length) { pushUndo(); S.xray.marks = []; saveMarks(); toast('Markings cleared. Press undo to bring them back.'); } break;
    case 'full': S.full = !S.full; break;
    case 'assist': S.assist = !S.assist; break;
    case 'closeAssist': S.assist = false; if (S.listening) await stopListening(); break;
    case 'handsfree': if (V.handsfree) await stopListening(); else { if (S.listening) await call('voice_stop').catch(() => {}); S.listening = false; await startListening('handsfree'); if (V.handsfree) speakText('Hands-free is on. Say Remi, then your command.'); } return;
    case 'speak': S.speak = !S.speak; if (!S.speak && window.speechSynthesis) speechSynthesis.cancel(); toast(S.speak ? 'Remi will read every answer aloud' : 'Remi will only speak answers to voice questions'); break;
    case 'mic': toggleMic(); return;
    case 'testVoice': speakText(VOICE_SAMPLE); return;
    case 'resetVoice': for (const k of Object.keys(VOICE)) delete VOICE[k]; saveVoice(); speakText(VOICE_SAMPLE); break;
    case 'menu': S.menu = S.menu ? false : true; break;
    case 'changeView': S.menu = 'view'; break;
    case 'unreview': S.menu = false; await guard(async () => { await call('mark_reviewed', x.id, !x.reviewed); x.reviewed = !x.reviewed; await reloadPatient(); }); return;
    case 'deleteXray': S.menu = false; S.delId = x.id; S.modal = { kind: 'confirm' }; break;
    case 'deleteTab': S.menu = false; S.delId = +el.dataset.del; S.modal = { kind: 'confirm' }; break;
    case 'confirmDelete': S.modal = null; await guard(async () => { const del = S.delId; await call('delete_xray', del); S.pdata = await call('get_patient', S.pid); const xs = allXrays(); if (del !== S.xid && xs.some(v => v.id === S.xid)) render(); else if (xs.length) await openXray(xs[0].id); else { S.xray = null; S.xid = null; render(); } toast('X-ray deleted'); }); return;
    case 'note': S.writing = true; render(); { const b = $('#noteBox'); if (b) b.focus(); } return;
    case 'cancelNote': S.writing = false; break;
    case 'saveNote': { const t = ($('#noteBox').value || '').trim(); S.writing = false; if (t) await guard(async () => { await call('add_note', S.pid, t); await reloadPatient(); toast("Note saved to today's visit"); }); render(); return; }
    case 'refer': S.modal = { kind: 'refer' }; render(); $('#r_to').focus(); return;
    case 'save': if (!x || x.view === 'unsure') return; await guard(async () => { await call('mark_reviewed', x.id, true); x.reviewed = true; await reloadPatient(); toast('Marked as reviewed and saved'); }); return;
    case 'export': S.busy = 'Making the PDF report…'; render(); await guard(async () => { const r = await call('export_report', S.pid); S.busy = ''; await reloadPatient(); toast(`PDF saved in the patient's files${r.download ? ' and Downloads' : ''}`); }); S.busy = ''; render(); return;
    case 'signOff': S.modal = { kind: 'signoff' }; break;
    case 'confirmSignOff': S.modal = null; S.busy = 'Signing off and writing the PDF report…'; render(); await guard(async () => { const r = await call('sign_off', S.pid); S.busy = ''; await reloadPatient(); if (S.xid) S.xray = await call('get_xray', S.xid); toast(`Signed off. PDF saved in the patient's files${r.download ? ' and Downloads' : ''}`); }); S.busy = ''; render(); return;
    case 'openSigned': { const v = latestVisit(); if (v && v.report) await guard(() => call('open_report', S.pid, v.report)); return; }
    case 'openReport': await guard(() => call('open_report', S.pid, el.dataset.file)); return;
    case 'account': await openAccount(); return;
    case 'closeAccount': S.screen = S.prevScreen || 'list'; if (S.screen === 'list') return goList(); break;
    case 'recheckModels': S.models = null; render(); await guard(async () => { S.models = await call('model_status'); }); break;
    case 'removeAssistant': await guard(async () => { await call('remove_assistant', el.dataset.id); S.acct = await call('account'); toast('Assistant account removed'); }); break;
    case 'openFolder': await guard(() => call('open_patient_folder', S.pid)); return;
    default: return;
  }
  render();
}

/* ---------- render + events ---------- */
async function openAccount() {
  await guard(async () => {
    if (S.screen !== 'account') S.prevScreen = S.screen;
    S.acct = await call('account'); S.screen = 'account'; S.models = null; render();
    S.models = await call('model_status'); if (S.screen === 'account') render();
  });
}
function voiceCard() {
  const vs = voiceList(), cur = pickVoice(), rate = VOICE.rate || 1, pitch = VOICE.pitch || 1;
  return `<div class="card acct"><h4>Remi's voice</h4>
    <p class="empty-note" style="margin:0">How Remi sounds when it answers out loud. These are the English voices installed in Windows; you can also say "Remi, change voice".</p>
    ${vs.length ? `<div class="field"><label for="v_name">Voice</label><select id="v_name">${vs.map(v => `<option value="${esc(v.name)}" ${cur && cur.name === v.name ? 'selected' : ''}>${esc(voiceLabel(v))}</option>`).join('')}</select></div>
    <div class="field"><label for="v_rate">Speed <span class="num" id="v_rate_val">${rate.toFixed(2)}×</span></label><input id="v_rate" type="range" min="0.7" max="1.4" step="0.05" value="${rate}"></div>
    <div class="field"><label for="v_pitch">Pitch <span class="num" id="v_pitch_val">${pitch.toFixed(2)}</span></label><input id="v_pitch" type="range" min="0.7" max="1.3" step="0.05" value="${pitch}"></div>
    <div class="actions"><button type="button" class="btn" data-act="resetVoice">Default</button><button type="button" class="btn solid" data-act="testVoice">${ICON.speaker}Hear it</button></div>`
    : '<p class="empty-note" style="margin:0">No English voices found on this computer. Add one in Windows Settings › Time &amp; language › Speech.</p>'}
  </div>`;
}
function accountScreen() {
  const a = S.acct, me = a.me, doc = me.role === 'doctor';
  const models = S.models;
  const modelRows = !models ? '<div class="empty-note">Checking every model…</div>' : models.map(m => `
    <div class="model-row">
      <span class="status ${m.ok ? 'ok' : m.loaded ? 'warn' : 'bad'}"></span>
      <div><b>${esc(m.name)}</b><span>${esc(m.what)}</span></div>
      <div class="m-view">${esc(m.view)}</div>
      <div class="m-score">${esc(m.score)}</div>
      <div class="m-state">${m.ok ? `Working · ${m.ms} ms` : esc(m.problem || 'Not working')}${m.key === 'bunion' && m.ok ? `<span>${m.finder ? 'Foot finder on' : 'Foot finder not installed'}</span>` : ''}</div>
    </div>`).join('');
  const okCount = models ? models.filter(m => m.ok).length : 0;
  return `<div class="list account">
    <div class="list-head">
      <div><div class="brand"><span class="logo"><img src="img/remi_mark.png" alt=""></span><b>Remi</b><span>AI</span></div><div class="eyebrow">${doc ? 'Doctor account' : 'Assistant account'}</div><h2>Account</h2></div>
      <div class="list-tools"><button class="btn" data-act="closeAccount">${ICON.left}Back</button><button class="btn solid" data-act="logout">Sign out</button></div>
    </div>
    <div class="acct-grid">
      <form class="card acct" id="acctForm"><h4>Your details</h4>
        <dl class="kv"><dt>${doc ? 'Doctor ID' : 'Assistant ID'}</dt><dd class="num">${esc(me.doctor_id)}</dd><dt>Role</dt><dd>${doc ? 'Doctor · can sign off' : 'Assistant · can make reports, not sign off'}</dd>
        ${!doc && a.supervisor ? `<dt>Works with</dt><dd>${esc(a.supervisor.name)} (${esc(a.supervisor.doctor_id)})</dd>` : ''}<dt>Since</dt><dd class="num">${fmtDate(me.created)}</dd></dl>
        <div class="field"><label for="a_name">Name</label><input id="a_name" value="${esc(me.name)}" required></div>
        <div class="field"><label for="a_clinic">Clinic</label><input id="a_clinic" value="${esc(me.clinic)}"></div>
        <div class="err" id="acctErr"></div><div class="actions"><button class="btn solid" type="submit">Save details</button></div>
      </form>
      <form class="card acct" id="pwForm"><h4>Change password</h4>
        <div class="field"><label for="pw_old">Current password</label><input id="pw_old" type="password" autocomplete="current-password" required></div>
        <div class="field"><label for="pw_new">New password</label><input id="pw_new" type="password" autocomplete="new-password" placeholder="At least 6 characters" required></div>
        <div class="field"><label for="pw_new2">New password again</label><input id="pw_new2" type="password" autocomplete="new-password" required></div>
        <div class="err" id="pwErr"></div><div class="actions"><button class="btn solid" type="submit">Change password</button></div>
      </form>
      ${voiceCard()}
      ${doc ? `<form class="card acct" id="asstForm"><h4>Assistant logins</h4>
        <p class="empty-note" style="margin:0">Assistants sign in with their own ID. They can add patients, X-rays and make the PDF report, but only you can sign off.</p>
        ${a.assistants.length ? `<div class="asst-list">${a.assistants.map(x => `<div><b>${esc(x.name)}</b><span class="num">${esc(x.doctor_id)}</span><button type="button" class="btn" data-act="removeAssistant" data-id="${esc(x.doctor_id)}">Remove</button></div>`).join('')}</div>` : ''}
        <div class="form-grid"><div class="field"><label for="as_id">Assistant ID</label><input id="as_id" placeholder="e.g. AS-01" required></div>
        <div class="field"><label for="as_name">Name</label><input id="as_name" required></div>
        <div class="field wide"><label for="as_pw">Password</label><input id="as_pw" type="password" autocomplete="new-password" placeholder="At least 6 characters" required></div></div>
        <div class="err" id="asErr"></div><div class="actions"><button class="btn solid" type="submit">Add assistant</button></div>
      </form>` : ''}
    </div>
    <div class="card models-card">
      <div class="models-head"><div><h4>AI models in Remi</h4><span>${models ? `${okCount} of ${models.length} working. Each one is test-run on this computer when you open this page.` : 'Test-running each model on this computer…'}</span></div>
        <button class="btn" data-act="recheckModels">Check again</button></div>
      <div class="model-row head"><span></span><div>Model</div><div class="m-view">X-ray view</div><div class="m-score">Accuracy in testing</div><div class="m-state">Status now</div></div>
      ${modelRows}
    </div>
    <p class="empty-note">Data folder: <span class="num">${esc(a.data_dir)}</span></p>
  </div>`;
}
function render() {
  const el = $('#screen');
  if (S.screen === 'loading') { el.innerHTML = '<div class="login"><div class="spinner"></div></div>'; return; }
  el.className = 'screen' + (S.screen === 'patient' ? ' patient-screen' : '');
  const chatScroll = $('#msgs') ? $('#msgs').scrollTop : null;
  el.innerHTML = { login: loginScreen, list: listScreen, patient: patientScreen, account: accountScreen }[S.screen]();
  if (S.modal) {
    const m = document.createElement('div');
    m.innerHTML = S.modal.kind === 'patient' ? patientForm(S.modal.edit ? S.pdata.patient : null, S.modal.edit ? S.pdata.history : null)
      : S.modal.kind === 'refer' ? referForm() : S.modal.kind === 'signoff' ? signOffForm() : confirmForm('Delete this X-ray?', 'confirmDelete');
    el.appendChild(m.firstElementChild);
  }
  if (S.screen !== 'login') el.insertAdjacentHTML('beforeend', handsfreePill() + floatingRemi());
  const msgs = $('#msgs'); if (msgs) msgs.scrollTop = chatScroll === null ? msgs.scrollHeight : msgs.scrollHeight;
}
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-open],[data-xray],[data-tab],[data-act],[data-chip],[data-say],[data-view]');
  if (!t) {
    if (S.mark.emenu && !e.target.closest('.er-menu')) { S.mark.emenu = false; render(); }
    if (S.menu === true && !e.target.closest('.menu')) { S.menu = false; render(); }
    return;
  }
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
    if (k === 'overlay') S.overlay = S.overlay ? 0 : 1;
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
      S.newDoctor = false; await goList();
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
  if (id === 'acctForm') {
    try { S.doctor = await call('update_account', $('#a_name').value, $('#a_clinic').value); S.acct = await call('account'); render(); toast('Details saved'); }
    catch (x) { $('#acctErr').textContent = x.message; }
  }
  if (id === 'pwForm') {
    if ($('#pw_new').value !== $('#pw_new2').value) { $('#pwErr').textContent = "The two new passwords don't match."; return; }
    try { await call('change_password', $('#pw_old').value, $('#pw_new').value); render(); toast('Password changed'); }
    catch (x) { $('#pwErr').textContent = x.message; }
  }
  if (id === 'asstForm') {
    try { const a = await call('add_assistant', $('#as_id').value, $('#as_name').value, $('#as_pw').value); S.acct = await call('account'); render(); toast(`${a.name} can now sign in with ID ${a.doctor_id}`); }
    catch (x) { $('#asErr').textContent = x.message; }
  }
  if (id === 'chatForm') { const v = $('#chatIn').value.trim(); if (v) say(v); }
});
document.addEventListener('input', (e) => {
  if (e.target.id === 'v_rate' || e.target.id === 'v_pitch') {
    const k = e.target.id === 'v_rate' ? 'rate' : 'pitch', val = +e.target.value;
    VOICE[k] = val; saveVoice();
    $('#' + e.target.id + '_val').textContent = k === 'rate' ? val.toFixed(2) + '×' : val.toFixed(2);
  }
  if (e.target.id === 'listSearch') {
    S.q = e.target.value;
    const q = S.q;
    guard(async () => { const rows = await call('list_patients', q); if (q !== S.q || S.screen !== 'list') return; S.patients = rows; render(); const i = $('#listSearch'); i.focus(); i.setSelectionRange(S.q.length, S.q.length); });
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.matches('tr[data-open]')) e.target.click();
  if (S.mark.on && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.target.matches('input, textarea')) { e.preventDefault(); doAct('mundo'); return; }
  if (e.key === 'F2' && S.doctor) { e.preventDefault(); toggleMic(); return; }
  if (e.key === 'Escape') { if (S.modal) { S.modal = null; render(); } else if (S.menu) { S.menu = false; render(); } }
  if (S.screen === 'patient' && !S.modal && !e.target.matches('input, textarea, select')) {
    if (e.key === 'ArrowRight' && S.xray) doAct('nextF');
    if (e.key === 'ArrowLeft' && S.xray) doAct('prevF');
  }
});
document.addEventListener('change', (e) => {
  if (e.target.id === 'v_name') { VOICE.name = e.target.value; saveVoice(); speakText(VOICE_SAMPLE); }
  if (e.target.id === 'v_rate' || e.target.id === 'v_pitch') speakText(VOICE_SAMPLE);
});
$('#fileIn').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; uploadFile(f); });
document.addEventListener('dragover', (e) => { if (S.screen === 'patient') { e.preventDefault(); const d = $('#drop'); if (d) d.classList.add('over'); } });
document.addEventListener('dragleave', () => { const d = $('#drop'); if (d) d.classList.remove('over'); });
document.addEventListener('drop', (e) => { if (S.screen !== 'patient') return; e.preventDefault(); uploadFile(e.dataTransfer.files[0]); });

(async function start() {
  render();
  try {
    S.info = await call('app_info');
    V.status = await call('voice_status').catch(() => null);
    if (V.status && V.status.downloading != null) { const t = setInterval(async () => { V.status = await call('voice_status').catch(() => V.status); updateVoiceUi(); if (!V.status || V.status.downloading == null) clearInterval(t); }, 1500); }
    S.screen = 'login';
    if (S.info.doctor) { S.doctor = S.info.doctor; await goList(); return; }
  } catch (e) { document.body.innerHTML = `<p style="padding:24px;color:#fff">Remi AI could not start: ${esc(e.message)}</p>`; return; }
  render();
})();

/* ---------- dragging the measuring points ---------- */
// The doctor can grab any point on the selected finding's lines. Angles update while dragging;
// on release Python recomputes and saves them (marked as adjusted by the doctor).
function acute(a0, a1, b0, b1) {
  const v1 = [a1[0] - a0[0], a1[1] - a0[1]], v2 = [b1[0] - b0[0], b1[1] - b0[1]];
  const n = Math.hypot(...v1) * Math.hypot(...v2); if (!n) return 0;
  const t = Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n))) * 180 / Math.PI;
  return Math.min(t, 180 - t);
}
function liveAngles(fd) {
  const r = fd.r;
  if (fd.key === 'bunion') { const p = r.points; r.hva = acute(p[0], p[1], p[2], p[3]); r.ima = acute(p[2], p[3], p[4], p[5]); }
  else { const [a, b] = [r.points.L10, r.points.L11], dx = b[0] - a[0]; r.angle = Math.atan2(b[1] - a[1], r.left ? -dx : dx) * 180 / Math.PI; }
}
let DRAG = null;
document.addEventListener('pointerdown', (e) => {
  const h = e.target.closest('.ov-grab'); if (!h || !S.xray || S.mark.on) return;
  const fd = findingList(S.xray)[S.f]; if (!fd) return;
  e.preventDefault(); e.stopPropagation();
  DRAG = { fd, id: h.dataset.pt, svg: h.ownerSVGElement, moved: false };
  document.body.classList.add('dragging');
}, true);
document.addEventListener('pointermove', (e) => {
  if (!DRAG) return;
  const m = DRAG.svg.getScreenCTM(); if (!m) return;
  const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
  const x = Math.max(0, Math.min(S.xray.width, pt.x)), y = Math.max(0, Math.min(S.xray.height, pt.y));
  const r = DRAG.fd.r;
  if (DRAG.fd.key === 'bunion') r.points[+DRAG.id] = [x, y]; else r.points[DRAG.id] = [x, y];
  liveAngles(DRAG.fd); DRAG.moved = true;
  const u = Math.max(S.xray.width, S.xray.height) / 900, g = DRAG.svg.querySelector('#ov');
  if (g) g.innerHTML = overlayAll(S.xray, DRAG.fd, u);
});
document.addEventListener('pointerup', async () => {
  if (!DRAG) return;
  const { fd, moved } = DRAG; DRAG = null; document.body.classList.remove('dragging');
  if (!moved) return;
  await guard(async () => {
    const pts = fd.key === 'bunion' ? fd.r.points : { L10: fd.r.points.L10, L11: fd.r.points.L11 };
    const r = await call('adjust_points', S.xid, fd.key, fd.i || 0, pts);
    if (fd.key === 'bunion') S.xray.results.checks.bunion.feet[fd.i] = r; else S.xray.results.checks.flat_foot = r;
    await reloadPatient();
  });
});
