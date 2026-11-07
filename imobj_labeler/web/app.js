import { BoxEditor, touch, isTyping } from "./canvas.js";

const $ = (id) => document.getElementById(id);
const PARAM_KEYS = ["channel", "flatten", "cmap", "lo", "hi"];

const S = {
  mode: "label",
  folder: null,
  out: null,
  files: [],              // [{name, registered}]
  classes: { classes: [] },
  curClass: 0,
  params: { channel: null, flatten: "plane", cmap: "viridis", lo: 1, hi: 99 }, // global defaults
  doc: null,              // current file: {name, anns, params, dirty, registered, undo, redo, snap, info}
  drafts: new Map(),      // name -> doc with unsaved edits
  model: null,            // {path, name}
  modelInfo: null,
  loadToken: 0,
};

// ---------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { "Content-Type": "application/json" };
    init.method = opts.method || "POST";
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let d = r.statusText;
    try { d = (await r.json()).detail || d; } catch { /* ignore */ }
    const err = new Error(typeof d === "string" ? d : JSON.stringify(d));
    err.status = r.status;
    throw err;
  }
  return r.json();
}
function qs(o) {
  return new URLSearchParams(Object.entries(o).filter(([, v]) => v !== null && v !== undefined && v !== "")).toString();
}
let msgTimer;
function msg(text, kind = "") {
  const el = $("st-msg");
  el.textContent = text; el.className = kind;
  clearTimeout(msgTimer);
  if (kind !== "err") msgTimer = setTimeout(() => { el.textContent = ""; }, 6000);
}
function fail(e) { console.error(e); msg(e.message || String(e), "err"); }
function clone(x) { return JSON.parse(JSON.stringify(x)); }

function classInfo(id) {
  const c = S.classes.classes.find((c) => c.id === id);
  return c ? c : { id, name: `cls ${id}?`, color: "#9aa0a6" };
}

// ---------------------------------------------------------------- editor
const editor = new BoxEditor($("canvas"), {
  onChange: () => commit(),
  onSelect: () => renderObjects(),
  onCursor: (p) => { $("st-pos").textContent = p ? `x ${p[0].toFixed(1)}  y ${p[1].toFixed(1)}` : ""; },
  getClass: () => S.curClass,
  classInfo,
});

function commit() {
  const d = S.doc;
  if (!d) return;
  d.undo.push(d.snap);
  if (d.undo.length > 200) d.undo.shift();
  d.redo = [];
  d.snap = JSON.stringify(d.anns);
  d.dirty = true;
  refreshDocUI();
}
function undo(redo = false) {
  const d = S.doc;
  if (!d) return;
  const from = redo ? d.redo : d.undo, to = redo ? d.undo : d.redo;
  if (!from.length) return;
  to.push(d.snap);
  d.snap = from.pop();
  d.anns = JSON.parse(d.snap);
  d.dirty = true;
  editor.setAnnotations(d.anns);
  editor.select(-1);
  refreshDocUI();
}

// ---------------------------------------------------------------- files
async function loadFiles(selectName) {
  const r = await api("/api/files");
  S.files = r.files;
  renderFiles();
  $("empty").textContent = S.files.length ? "" : "No supported files (.sxm, .png, .jpg, .tif, .bmp) in this folder.";
  $("empty").classList.toggle("hidden", !!S.files.length);
  const target = selectName || (S.files.find((f) => !f.registered) || S.files[0] || {}).name;
  if (target) await openFile(target);
}

function visibleFiles() {
  const q = $("file-filter").value.trim().toLowerCase(), show = $("file-show").value;
  return S.files.filter((f) => (!q || f.name.toLowerCase().includes(q)) &&
    (show === "all" || (show === "done") === f.registered));
}

function renderFiles() {
  const ul = $("file-list");
  ul.innerHTML = "";
  for (const f of visibleFiles()) {
    const li = document.createElement("li");
    li.textContent = f.name;
    li.title = f.name;
    li.dataset.name = f.name;
    li.classList.toggle("done", f.registered);
    li.classList.toggle("draft", S.drafts.has(f.name) || (S.doc?.name === f.name && S.doc.dirty));
    li.classList.toggle("active", S.doc?.name === f.name);
    li.onclick = () => openFile(f.name).catch(fail);
    ul.appendChild(li);
  }
  const n = S.files.filter((f) => f.registered).length;
  $("file-count").textContent = S.files.length ? `${n} / ${S.files.length} registered` : "";
  ul.querySelector("li.active")?.scrollIntoView({ block: "nearest" });
}

function stashDoc() {
  const d = S.doc;
  if (!d) return;
  if (d.dirty) S.drafts.set(d.name, d); else S.drafts.delete(d.name);
}

async function openFile(name, { keepView = false, reloadImageOnly = false } = {}) {
  const token = ++S.loadToken;
  let d;
  if (reloadImageOnly && S.doc?.name === name) {
    d = S.doc;
  } else {
    stashDoc();
    d = S.drafts.get(name);
    const q = d ? { name, ...d.params, use_registered: false } : { name, ...S.params, use_registered: true };
    const fi = await api(`/api/file?${qs(q)}`);
    if (token !== S.loadToken) return;
    const isNew = !d;
    if (!d) {
      d = { name, anns: fi.labels, params: fi.params, dirty: false, undo: [], redo: [], snap: "" };
      d.snap = JSON.stringify(d.anns);
    }
    d.registered = fi.registered;
    d.stem = fi.stem;
    d.versions = fi.versions || [];
    d.hasExisting = fi.has_existing;
    d.info = fi.info;
    d.usedRegistered = fi.used_registered_params;
    d.params.channel = fi.params.channel;
    S.doc = d;
    if (isNew && S.mode === "predict" && !fi.registered && !fi.labels.length && $("m-auto").checked && S.model) {
      d._autoPredict = true;
    }
  }
  setParamControls(d);
  const img = await loadImage(`/api/image?${qs({ name, ...d.params })}`);
  if (token !== S.loadToken) return;
  editor.setImage(img, keepView || reloadImageOnly);
  editor.setAnnotations(d.anns);
  if (!reloadImageOnly) editor.select(-1);
  $("st-size").textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
  refreshDocUI();
  if (d._autoPredict) { d._autoPredict = false; await predict(); }
}

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = async () => {
      try { const r = await fetch(src); const j = await r.json(); rej(new Error(j.detail || "image load failed")); }
      catch { rej(new Error("image load failed")); }
    };
    im.src = src;
  });
}

function step(delta) {
  const list = visibleFiles();
  if (!list.length) return;
  const i = list.findIndex((f) => f.name === S.doc?.name);
  const j = i < 0 ? 0 : Math.min(Math.max(i + delta, 0), list.length - 1);
  if (list[j] && list[j].name !== S.doc?.name) openFile(list[j].name).catch(fail);
}

// ---------------------------------------------------------------- doc UI
function refreshDocUI() {
  const d = S.doc;
  $("cur-name").textContent = d ? d.name : "—";
  const chip = $("cur-status");
  chip.className = "chip";
  if (!d) chip.textContent = "";
  else if (d.dirty) { chip.textContent = d.registered ? "modified" : "unsaved"; chip.classList.add("dirty"); }
  else if (d.registered) {
    const n = d.versions?.length || 1;
    chip.textContent = n > 1 ? `registered · ${d.stem} (${n} versions)` : "registered";
    chip.classList.add("done");
  }
  else chip.textContent = d.anns.some((a) => a.source === "existing") ? "existing labels" : "new";
  if (d?.registered && d.versions?.length > 1) chip.title = `Versions: ${d.versions.join(", ")}\nShowing the latest: ${d.stem}`;
  else chip.title = "";
  $("btn-unregister").classList.toggle("hidden", !d?.registered);
  $("btn-unregister").textContent = d?.versions?.length > 1 ? `unregister this version (${d.stem})` : "unregister this file";
  $("btn-revert").classList.toggle("hidden", !(d?.registered && d.hasExisting));
  renderObjects();
  renderFiles();
}

function setParamControls(d) {
  const sel = $("p-channel");
  sel.innerHTML = "";
  const chans = d.info?.channels || [];
  for (const c of chans) sel.add(new Option(c, c));
  sel.disabled = !chans.length;
  if (!chans.length) sel.add(new Option("RGB image", ""));
  sel.value = d.params.channel || "";
  for (const id of ["p-flatten", "p-cmap", "p-lo", "p-hi"]) $(id).disabled = d.info?.kind === "rgb";
  $("p-flatten").value = d.params.flatten;
  $("p-cmap").value = d.params.cmap;
  $("p-lo").value = d.params.lo;
  $("p-hi").value = d.params.hi;
  const notes = [];
  if (d.info?.kind === "rgb") notes.push("RGB image: shown and saved as is.");
  if (d.info?.fallback) notes.push("Requested channel missing: using the first channel.");
  if (d.usedRegistered) notes.push("Showing the settings used when this file was registered.");
  $("p-note").textContent = notes.join(" ");
}

function onParamChange() {
  const d = S.doc;
  if (!d) return;
  const p = {
    channel: $("p-channel").value || null, flatten: $("p-flatten").value, cmap: $("p-cmap").value,
    lo: parseFloat($("p-lo").value), hi: parseFloat($("p-hi").value),
  };
  if (!(p.lo >= 0 && p.hi <= 100 && p.lo < p.hi)) { msg("Clip needs 0 ≤ low < high ≤ 100", "err"); return; }
  d.params = p;
  Object.assign(S.params, p);
  d.usedRegistered = false;
  if (d.registered) { d.dirty = true; }
  openFile(d.name, { reloadImageOnly: true }).catch(fail);
}

// ---------------------------------------------------------------- classes
async function loadClasses() {
  S.classes = await api("/api/classes");
  if (!S.classes.classes.some((c) => c.id === S.curClass)) S.curClass = S.classes.classes[0]?.id ?? 0;
  renderClasses();
  editor.draw();
  renderObjects();
}

function renderClasses() {
  const box = $("class-list");
  box.innerHTML = "";
  S.classes.classes.forEach((c, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="swatch" style="background:${c.color}"></span>${esc(c.name)}` +
      (i < 9 ? ` <span class="kbd">${i + 1}</span>` : "");
    b.classList.toggle("active", c.id === S.curClass);
    b.onclick = () => setClass(c.id);
    box.appendChild(b);
  });
}

function setClass(id) {
  S.curClass = id;
  renderClasses();
  const a = S.doc?.anns[editor.sel];
  if (a && a.cls !== id) { a.cls = id; touch(a); commit(); editor.draw(); }
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------------------------------------------------------------- objects
// Display order of the object list (indices into d.anns); the annotations themselves are not reordered.
// conf: highest first, boxes without a confidence (manual) last.
// center / origin: distance of the box center, in normalized image coords, to (0.5, 0.5) / (0, 0) = top-left.
function objectOrder() {
  const d = S.doc;
  if (!d) return [];
  const W = editor.width || 1, H = editor.height || 1;
  const mode = $("obj-sort").value;
  const key = (a) => {
    if (mode === "conf") return a.conf == null ? Infinity : -a.conf;
    const cx = (a.x1 + a.x2) / 2 / W, cy = (a.y1 + a.y2) / 2 / H;
    const [tx, ty] = mode === "center" ? [0.5, 0.5] : [0, 0];
    return Math.hypot(cx - tx, cy - ty);
  };
  return d.anns.map((a, i) => [key(a), i]).sort((p, q) => p[0] - q[0] || p[1] - q[1]).map((p) => p[1]);
}

function renderObjects() {
  const ul = $("obj-list"), d = S.doc;
  ul.innerHTML = "";
  if (!d) { $("obj-count").textContent = ""; return; }
  $("obj-count").textContent = `(${d.anns.length})`;
  objectOrder().forEach((i) => {
    const a = d.anns[i];
    const li = document.createElement("li");
    li.classList.toggle("sel", i === editor.sel);
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = classInfo(a.cls).color;
    const s = document.createElement("select");
    const ids = new Set(S.classes.classes.map((c) => c.id));
    for (const c of S.classes.classes) s.add(new Option(c.name, c.id));
    if (!ids.has(a.cls)) s.add(new Option(`cls ${a.cls}?`, a.cls));
    s.value = a.cls;
    s.onclick = (e) => e.stopPropagation();
    s.onchange = () => { a.cls = parseInt(s.value, 10); touch(a); commit(); editor.draw(); };
    const src = document.createElement("span");
    src.className = "src" + (a.source === "model" ? " model" : "");
    src.textContent = `${a.source}${a.conf != null ? " · " + a.conf.toFixed(2) : ""} · ${Math.round(a.x2 - a.x1)}×${Math.round(a.y2 - a.y1)}`;
    const x = document.createElement("button");
    x.className = "x"; x.textContent = "✕"; x.title = "Delete";
    x.onclick = (e) => { e.stopPropagation(); deleteAnn(i); };
    li.append(sw, s, src, x);
    li.onclick = () => editor.select(i);
    ul.appendChild(li);
  });
  ul.querySelector("li.sel")?.scrollIntoView({ block: "nearest" });
}

function deleteAnn(i) {
  const d = S.doc;
  if (!d || i < 0 || i >= d.anns.length) return;
  d.anns.splice(i, 1);
  editor.select(-1);
  editor.setAnnotations(d.anns);
  commit();
}

// ---------------------------------------------------------------- register
function sourceSummary(anns) {
  const s = [...new Set(anns.map((a) => a.source || "manual"))].sort();
  return s.length ? s.join("+") : "empty";
}

async function register() {
  const d = S.doc;
  if (!d) return;
  if (!d.anns.length && !confirm(`Register "${d.name}" with NO objects (negative example)?`)) return;
  try {
    const res = await api("/api/register", { json: {
      name: d.name, params: d.params, annotations: d.anns,
      source: sourceSummary(d.anns),
      overwrite: $("chk-overwrite").checked,
      model: d.anns.some((a) => a.source === "model" || a.source === "edited") ? (S.model?.name || "") : "",
    } });
    d.dirty = false;
    d.registered = true;
    d.usedRegistered = false;
    d.stem = res.stem;
    d.versions = res.versions;
    S.drafts.delete(d.name);
    const f = S.files.find((f) => f.name === d.name);
    if (f) f.registered = true;
    const how = res.replaced ? "overwritten" : res.versions.length > 1 ? "new copy" : "saved";
    let m = `Registered ${res.stem} (${how}): ${res.n_objects} object(s)`;
    if (res.unknown_classes.length) m += ` — class id(s) ${res.unknown_classes.join(", ")} not in classes.json`;
    msg(m, res.unknown_classes.length ? "err" : "ok");
    $("out-path").textContent = S.out || "";
    if ($("chk-advance").checked) {
      const list = visibleFiles();
      const i = list.findIndex((x) => x.name === d.name);
      const next = list.slice(i + 1).find((x) => !x.registered) || list[i + 1];
      if (next) { await openFile(next.name); return; }
    }
    refreshDocUI();
  } catch (e) { fail(e); }
}

async function removeRegistered(all) {
  const d = S.doc;
  if (!d?.registered) return;
  const n = d.versions?.length || 1;
  const question = all
    ? `Delete all ${n} registered version${n > 1 ? "s" : ""} of "${d.name}" from obj_det_labels and show the existing labels from ../labels again?`
    : `Remove the registered image and labels for ${n > 1 ? `version "${d.stem}" of "${d.name}"` : `"${d.name}"`}?`;
  if (!confirm(question)) return;
  try {
    await api(`/api/register?${qs({ name: d.name, all })}`, { method: "DELETE" });
    S.drafts.delete(d.name);
    S.doc = null;
    await openFile(d.name);
    const f = S.files.find((f) => f.name === d.name);
    if (f) f.registered = !!S.doc?.registered;
    renderFiles();
    msg(all ? `Reverted ${d.name} to existing labels` : "Unregistered " + d.name, "ok");
  } catch (e) { fail(e); }
}
const unregister = () => removeRegistered(false);
const revertToExisting = () => removeRegistered(true);

// ---------------------------------------------------------------- dataset (data.yaml)
let dsSummary = null;
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

async function openDatasetDialog() {
  $("ds-msg").textContent = "";
  $("ds-counts").textContent = "";
  $("dlg-dataset").showModal();
  try {
    dsSummary = await api("/api/dataset");
    const s = dsSummary.settings;
    // a never-split folder shows the 0.2 suggestion; a saved split shows what was used
    if (dsSummary.yaml_exists && s.val_fraction > 0) $("ds-val").value = s.val_fraction;
    $("ds-seed").value = s.seed;
    $("ds-source").value = s.source;
    $("ds-source").querySelector('[value="existing"]').disabled = !dsSummary.has_existing;
    showDatasetCounts();
  } catch (e) { dsSummary = null; $("ds-msg").textContent = e.message; }
}

function showDatasetCounts() {
  if (!dsSummary) return;
  const c = dsSummary.counts[$("ds-source").value];
  let t = `${plural(c.files, "source file")}, ${plural(c.images, "image")}`;
  if (c.overridden) t += ` · ${c.overridden} existing replaced by registered`;
  if ($("ds-source").value === "existing" && !dsSummary.has_existing)
    t = "No existing labels: the data folder must be an images/ folder with a ../labels folder.";
  $("ds-counts").textContent = t;
}

async function writeDataset() {
  try {
    const r = await api("/api/data_yaml", { json: {
      source: $("ds-source").value,
      val_fraction: parseFloat($("ds-val").value) || 0,
      seed: parseInt($("ds-seed").value, 10) || 0 } });
    $("ds-msg").textContent = r.split
      ? `Wrote ${r.path}: ${r.n_train} train / ${r.n_val} val images (${r.groups_train} / ${r.groups_val} source files)`
      : `Wrote ${r.path}: all ${r.n_train} images used for both train and val`;
  } catch (e) { $("ds-msg").textContent = e.message; }
}

// ---------------------------------------------------------------- prediction
async function scanModels(folder) {
  const r = await api(`/api/models?${qs({ folder })}`);
  $("m-folder").value = r.folder || "";
  const sel = $("m-select");
  sel.innerHTML = "";
  if (!r.models.length) { sel.add(new Option(r.folder ? "No .pt files found" : "Set a models folder", "")); return; }
  for (const m of r.models) sel.add(new Option(m.name, m.path));
  const prev = localStorage.getItem("imobj.model");
  if (prev && r.models.some((m) => m.path === prev)) sel.value = prev;
  await selectModel();
}

async function selectModel() {
  const sel = $("m-select");
  const path = sel.value;
  S.model = path ? { path, name: sel.selectedOptions[0].textContent } : null;
  S.modelInfo = null;
  $("m-info").textContent = ""; $("m-warn").classList.add("hidden");
  if (!path) return;
  try { localStorage.setItem("imobj.model", path); } catch { /* ignore */ }
  $("m-info").textContent = "Loading model…";
  try {
    const info = await api(`/api/model_info?${qs({ path })}`);
    S.modelInfo = info;
    const names = Object.entries(info.names).map(([k, v]) => `${k}: ${v}`).join(", ");
    $("m-info").textContent = `${info.task} · imgsz ${info.imgsz} · classes {${names}}`;
    $("m-imgsz").placeholder = `model default (${info.imgsz})`;
    const mm = info.class_mismatch || {};
    if (await adoptModelClasses()) { /* classes.json created from the model */ }
    else if (Object.keys(mm).length) {
      const w = $("m-warn");
      w.innerHTML = `Model classes differ from classes.json (${Object.entries(mm).map(([k, v]) => `${k}: ${esc(v)}`).join(", ")}). ` +
        `<button class="link" id="m-adopt">use model classes</button>`;
      w.classList.remove("hidden");
      $("m-adopt").onclick = async () => {
        if (!confirm("Replace classes.json with the model's class names?")) return;
        await api(`/api/classes/from_model?${qs({ path })}`, { method: "POST" }).catch(fail);
        await loadClasses(); w.classList.add("hidden");
      };
    }
  } catch (e) { $("m-info").textContent = ""; fail(e); }
}

// In Prediction mode, a project without classes.json takes the model's class names.
async function adoptModelClasses() {
  if (S.mode !== "predict" || !S.folder || !S.model || !S.modelInfo || S.classes.exists) return false;
  try {
    await api(`/api/classes/from_model?${qs({ path: S.model.path })}`, { method: "POST" });
    S.modelInfo.classes_file_exists = true;
    $("m-warn").classList.add("hidden");
    await loadClasses();
    msg("classes.json created from the model's class names", "ok");
    return true;
  } catch (e) { fail(e); return false; }
}

let predicting = false;
async function predict() {
  const d = S.doc;
  if (!d) return;
  if (!S.model) { msg("Select a model first (Prediction panel)", "err"); return; }
  if (predicting) return;
  predicting = true;
  $("m-run").disabled = true;
  msg("Predicting…");
  try {
    const imgsz = parseInt($("m-imgsz").value, 10) || null;
    const r = await api("/api/predict", { json: {
      name: d.name, params: d.params, model: S.model.path,
      conf: parseFloat($("m-conf").value), iou: parseFloat($("m-iou").value), imgsz,
    } });
    if (S.doc !== d) return;
    const keep = d.anns.filter((a) => a.source === "manual");
    d.anns.splice(0, d.anns.length, ...keep, ...r.annotations);
    editor.select(-1);
    editor.setAnnotations(d.anns);
    commit();
    msg(`${r.annotations.length} object(s) predicted` + (keep.length ? `, ${keep.length} manual kept` : ""), "ok");
  } catch (e) { fail(e); } finally { predicting = false; $("m-run").disabled = false; }
}

let confTimer;
function onThresholdChange() {
  $("m-conf-v").textContent = parseFloat($("m-conf").value).toFixed(2);
  $("m-iou-v").textContent = parseFloat($("m-iou").value).toFixed(2);
  const d = S.doc;
  // live re-predict only while the predictions are still unreviewed
  if (S.mode !== "predict" || !d || !S.model) return;
  if (!d.anns.some((a) => a.source === "model") || d.anns.some((a) => a.source === "edited")) return;
  clearTimeout(confTimer);
  confTimer = setTimeout(predict, 300);
}

async function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("predict-panel").classList.toggle("hidden", mode !== "predict");
  try { localStorage.setItem("imobj.mode", mode); } catch { /* ignore */ }
  if (mode !== "predict") return;
  await adoptModelClasses();
  const d = S.doc;
  if (d && !d.dirty && !d.registered && $("m-auto").checked && S.model && !d.anns.length) await predict();
}

// ---------------------------------------------------------------- dialogs
// Folder selection: the native OS dialog (opened by the local server) when available,
// otherwise the in-app picker with Places / Recent / Drives.
async function chooseFolder({ title, initial = "", purpose = "data" }) {
  msg("Folder dialog open (if you can't see it, check the taskbar)");
  try {
    const r = await api("/api/pick_folder", { json: { initial, title, purpose } });
    msg("");
    return r.path;
  } catch (e) {
    msg("");
    if (e.status === 501) return inAppPicker({ title, initial, purpose });
    fail(e);
    return null;
  }
}

const picker = { resolve: null, purpose: "data", path: "" };
const PLACE_ICON = { home: "🏠", desktop: "🖥️", documents: "📄", downloads: "⬇️", pictures: "🖼️",
  cloud: "☁️", drive: "💽", recent: "🕘", models: "🧠" };

async function inAppPicker({ title, initial, purpose }) {
  picker.purpose = purpose;
  $("fb-title").textContent = title;
  $("fb-open").textContent = purpose === "models" ? "Use this models folder" : "Open this folder";
  try {
    renderPlaces(await api("/api/places"));
    await browse(initial || "");
  } catch (e) { fail(e); return null; }
  $("dlg-folder").showModal();
  return new Promise((res) => { picker.resolve = res; });
}

function finishPicker(path) {
  const res = picker.resolve;
  picker.resolve = null;
  if ($("dlg-folder").open) $("dlg-folder").close();
  res?.(path);
}

function renderPlaces({ places, recent, models_folder }) {
  const nav = $("fb-places");
  nav.innerHTML = "";
  const group = (label, items) => {
    if (!items.length) return;
    const h = document.createElement("div");
    h.className = "group"; h.textContent = label;
    nav.appendChild(h);
    for (const it of items) {
      const b = document.createElement("button");
      b.dataset.path = it.path;
      b.title = it.path;
      b.innerHTML = `<span class="ico">${PLACE_ICON[it.kind] || "📁"}</span><span>${esc(it.name)}</span>`;
      b.onclick = () => browse(it.path);
      nav.appendChild(b);
    }
  };
  const baseName = (p) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;
  group("Places", places.filter((p) => p.kind !== "drive"));
  const rec = recent.map((p) => ({ name: baseName(p), path: p, kind: "recent" }));
  if (picker.purpose === "models" && models_folder)
    rec.unshift({ name: baseName(models_folder), path: models_folder, kind: "models" });
  group("Recent", rec);
  group("Drives", places.filter((p) => p.kind === "drive"));
}

function joinPath(base, name) {
  const sep = base.includes("\\") ? "\\" : "/";
  return base.replace(/[\\/]+$/, "") + sep + name;
}

async function browse(path) {
  try {
    const r = await api(`/api/browse?${qs({ path })}`);
    picker.path = r.path;
    $("fb-path").value = r.path;
    $("fb-up").disabled = !r.parent;
    $("fb-up").onclick = () => r.parent && browse(r.parent);

    // breadcrumb: C:\ › Users › ggn
    const crumbs = $("fb-crumbs");
    crumbs.innerHTML = "";
    const win = r.path.includes("\\");
    const parts = r.path.split(/[\\/]/).filter(Boolean);
    let acc = win ? "" : "/";
    parts.forEach((part, i) => {
      acc = i === 0 ? (win ? part + "\\" : "/" + part) : joinPath(acc, part);
      const target = acc;
      if (i > 0 || !win) { const s = document.createElement("span"); s.className = "sep"; s.textContent = "›"; crumbs.appendChild(s); }
      const b = document.createElement("button");
      b.textContent = part;
      b.onclick = () => browse(target);
      crumbs.appendChild(b);
    });

    const ul = $("fb-list");
    ul.innerHTML = "";
    for (const dname of r.dirs) {
      const li = document.createElement("li");
      li.innerHTML = `<span>📁</span><span>${esc(dname)}</span>`;
      li.onclick = () => browse(joinPath(r.path, dname));
      ul.appendChild(li);
    }
    if (!r.dirs.length) {
      const li = document.createElement("li");
      li.className = "empty"; li.textContent = "No subfolders";
      ul.appendChild(li);
    }

    const cnt = $("fb-count");
    if (picker.purpose === "models") {
      cnt.textContent = r.n_models ? `${r.n_models} .pt model(s) here` : "Subfolders are searched for .pt models";
      cnt.classList.toggle("has", r.n_models > 0);
    } else {
      cnt.textContent = r.n_files ? `${r.n_files} supported file(s) here` : "No supported files here";
      cnt.classList.toggle("has", r.n_files > 0);
    }
    document.querySelectorAll("#fb-places button").forEach((b) =>
      b.classList.toggle("active", b.dataset.path.replace(/[\\/]+$/, "").toLowerCase() === r.path.replace(/[\\/]+$/, "").toLowerCase()));
  } catch (e) { fail(e); }
}

async function pickDataFolder() {
  const p = await chooseFolder({ title: "Select data folder", initial: S.folder || "", purpose: "data" });
  if (p) await openFolder(p);
}

async function pickModelsFolder() {
  const p = await chooseFolder({ title: "Select models folder", initial: $("m-folder").value.trim(), purpose: "models" });
  if (p) await scanModels(p).catch(fail);
}

async function openFolder(path) {
  if ([...S.drafts.values()].some((d) => d.dirty) || S.doc?.dirty) {
    if (!confirm("Unsaved edits in this folder will be lost. Continue?")) return;
  }
  try {
    const st = await api("/api/open", { json: { folder: path } });
    $("dlg-folder").close();
    S.drafts.clear(); S.doc = null;
    applyState(st);
    await loadClasses();
    await loadFiles();
    if (S.model) await selectModel();
  } catch (e) { fail(e); }
}

function applyState(st) {
  S.folder = st.folder; S.out = st.out;
  $("folder-path").textContent = st.folder || "No folder open";
  $("folder-path").title = st.folder || "";
  $("out-path").textContent = st.out ? `→ ${st.out}` : "";
  $("out-path").title = st.out || "";
}

function openClassesDialog() {
  const body = $("cls-body");
  body.innerHTML = "";
  const addRow = (c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><input type="number" min="0" value="${c.id}"></td>
      <td><input type="text" value="${esc(c.name)}"></td>
      <td><input type="color" value="${c.color}"></td>
      <td><button class="link danger">remove</button></td>`;
    tr.querySelector("button").onclick = () => tr.remove();
    body.appendChild(tr);
  };
  S.classes.classes.forEach(addRow);
  $("cls-add").onclick = () => {
    const ids = [...body.querySelectorAll("tr")].map((tr) => parseInt(tr.querySelector("input").value, 10) || 0);
    const id = ids.length ? Math.max(...ids) + 1 : 0;
    const pal = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"];
    addRow({ id, name: "", color: pal[id % pal.length] });
  };
  $("cls-err").textContent = "";
  $("dlg-classes").showModal();
}

async function saveClasses() {
  if (!S.folder) { $("cls-err").textContent = "Open a data folder first."; return; }
  const classes = [...$("cls-body").querySelectorAll("tr")].map((tr) => {
    const [id, name, color] = tr.querySelectorAll("input");
    return { id: parseInt(id.value, 10), name: name.value.trim(), color: color.value };
  });
  try {
    await api("/api/classes", { method: "PUT", json: { task: "detect", classes } });
    $("dlg-classes").close();
    await loadClasses();
    msg("Classes saved", "ok");
  } catch (e) { $("cls-err").textContent = e.message; }
}

// ---------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  if (document.querySelector("dialog[open]") || isTyping(e)) return;
  const k = e.key, ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && k.toLowerCase() === "s") { e.preventDefault(); register(); return; }
  if (ctrl && k.toLowerCase() === "z") { e.preventDefault(); undo(e.shiftKey); return; }
  if (ctrl && k.toLowerCase() === "y") { e.preventDefault(); undo(true); return; }
  if (ctrl) return;
  if (k === "Enter") { e.preventDefault(); register(); }
  else if (/^[1-9]$/.test(k)) { const c = S.classes.classes[parseInt(k, 10) - 1]; if (c) setClass(c.id); }
  else if (k === "Delete" || k === "Backspace") { e.preventDefault(); deleteAnn(editor.sel); }
  else if (k === "Escape") editor.select(-1);
  else if (k === "a" || k === "A" || k === "PageUp") step(-1);
  else if (k === "d" || k === "D" || k === "PageDown") step(1);
  else if (k === "f" || k === "F") editor.fit();
  else if (k === "h" || k === "H") { $("chk-show").checked = !$("chk-show").checked; editor.showBoxes = $("chk-show").checked; editor.draw(); }
  else if (k === "p" || k === "P") { if (S.mode !== "predict") setMode("predict"); predict(); }
  else if (k === "Tab") {
    e.preventDefault();
    const order = objectOrder(), n = order.length;
    if (n) {
      const pos = order.indexOf(editor.sel);
      const next = pos < 0 ? (e.shiftKey ? n - 1 : 0) : ((pos + (e.shiftKey ? -1 : 1)) % n + n) % n;
      editor.select(order[next]);
    }
  } else if (k.startsWith("Arrow")) {
    e.preventDefault();
    if (editor.sel >= 0) {
      const s = e.shiftKey ? 10 : 1;
      editor.nudge(k === "ArrowLeft" ? -s : k === "ArrowRight" ? s : 0, k === "ArrowUp" ? -s : k === "ArrowDown" ? s : 0);
    } else step(k === "ArrowLeft" || k === "ArrowUp" ? -1 : 1);
  }
});

// ---------------------------------------------------------------- wiring
function wire() {
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("btn-open").onclick = pickDataFolder;
  $("fb-path").onkeydown = (e) => { if (e.key === "Enter") browse($("fb-path").value.trim()); };
  $("fb-cancel").onclick = () => finishPicker(null);
  $("fb-open").onclick = () => finishPicker(picker.path);
  $("dlg-folder").addEventListener("close", () => finishPicker(null));
  $("btn-classes").onclick = openClassesDialog;
  $("cls-cancel").onclick = () => $("dlg-classes").close();
  $("cls-save").onclick = saveClasses;
  $("btn-dataset").onclick = openDatasetDialog;
  $("ds-close").onclick = () => $("dlg-dataset").close();
  $("ds-source").onchange = showDatasetCounts;
  $("ds-build").onclick = writeDataset;
  $("btn-help").onclick = () => $("dlg-help").showModal();
  $("help-close").onclick = () => $("dlg-help").close();

  $("file-filter").oninput = renderFiles;
  $("file-show").onchange = renderFiles;
  $("btn-prev").onclick = () => step(-1);
  $("btn-next").onclick = () => step(1);
  $("btn-fit").onclick = () => editor.fit();
  $("chk-show").onchange = (e) => { editor.showBoxes = e.target.checked; editor.draw(); };
  $("chk-labels").onchange = (e) => { editor.showTags = e.target.checked; editor.draw(); };
  for (const id of ["p-channel", "p-flatten", "p-cmap", "p-lo", "p-hi"]) $(id).onchange = onParamChange;
  $("btn-clear").onclick = () => {
    const d = S.doc;
    if (!d?.anns.length || !confirm(`Remove all ${d.anns.length} boxes from this image?`)) return;
    d.anns.splice(0); editor.select(-1); editor.setAnnotations(d.anns); commit();
  };
  const savedSort = recall("imobj.objSort");
  if (savedSort && [...$("obj-sort").options].some((o) => o.value === savedSort)) $("obj-sort").value = savedSort;
  $("obj-sort").onchange = () => { store("imobj.objSort", $("obj-sort").value); renderObjects(); };
  if (recall("imobj.overwrite") === "0") $("chk-overwrite").checked = false;
  $("chk-overwrite").onchange = () => store("imobj.overwrite", $("chk-overwrite").checked ? "1" : "0");
  $("btn-register").onclick = register;
  $("btn-unregister").onclick = unregister;
  $("btn-revert").onclick = revertToExisting;

  $("m-browse").onclick = pickModelsFolder;
  $("m-folder").onkeydown = (e) => { if (e.key === "Enter") scanModels($("m-folder").value.trim()).catch(fail); };
  $("m-select").onchange = selectModel;
  $("m-conf").oninput = onThresholdChange;
  $("m-iou").oninput = onThresholdChange;
  $("m-run").onclick = predict;

  window.addEventListener("beforeunload", (e) => {
    if (S.doc?.dirty || S.drafts.size) { e.preventDefault(); e.returnValue = ""; }
  });
}

// ---------------------------------------------------------------- theme & layout
function store(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } }
function recall(k) { try { return localStorage.getItem(k); } catch { return null; } }

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("btn-theme").textContent = t === "light" ? "Theme: Light" : "Theme: Dark";
}

const PANEL_DEFAULT = { left: 240, right: 290 };
const PANEL_MIN = 160;

function setPanelWidth(side, w) {
  const max = Math.max(PANEL_MIN, window.innerWidth * 0.45);
  w = Math.round(Math.min(Math.max(w, PANEL_MIN), max));
  document.documentElement.style.setProperty(`--${side}-w`, `${w}px`);
  return w;
}

function wireResizers() {
  for (const side of ["left", "right"]) {
    const saved = parseInt(recall(`imobj.${side}W`), 10);
    if (saved) setPanelWidth(side, saved);
  }
  document.querySelectorAll(".resizer").forEach((el) => {
    const side = el.dataset.side;
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startW = document.querySelector(`aside.${side}`).getBoundingClientRect().width;
      el.classList.add("active");
      document.body.classList.add("resizing");
      const move = (ev) => {
        const dx = ev.clientX - startX;
        setPanelWidth(side, side === "left" ? startW + dx : startW - dx);
      };
      const up = () => {
        el.classList.remove("active");
        document.body.classList.remove("resizing");
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        store(`imobj.${side}W`, document.querySelector(`aside.${side}`).getBoundingClientRect().width);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
    });
    el.addEventListener("dblclick", () => store(`imobj.${side}W`, setPanelWidth(side, PANEL_DEFAULT[side])));
  });
}

async function init() {
  applyTheme(recall("imobj.theme") === "light" ? "light" : "dark");
  $("btn-theme").onclick = () => {
    const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(t);
    store("imobj.theme", t);
  };
  wireResizers();
  wire();
  try {
    const st = await api("/api/state");
    for (const m of st.flatten_modes) $("p-flatten").add(new Option(m, m));
    for (const c of st.colormaps) $("p-cmap").add(new Option(c, c));
    applyState(st);
    await loadClasses();
    let mode = "label";
    try { mode = localStorage.getItem("imobj.mode") || "label"; } catch { /* ignore */ }
    S.mode = mode;
    if (st.folder) await loadFiles();
    else $("empty").textContent = "Click “Open folder…” to choose a data folder.";
    if (st.settings.models_folder) await scanModels(st.settings.models_folder).catch(fail);
    await setMode(mode);
  } catch (e) { fail(e); }
}

init();
