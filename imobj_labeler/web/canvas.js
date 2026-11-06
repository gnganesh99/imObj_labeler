// Image viewer + bounding-box editor on a <canvas>.
// Annotations are in image pixel coordinates (top-left origin).

const HANDLE = 6;      // screen px
const MIN_BOX = 2;     // image px

export class BoxEditor {
  constructor(canvas, { onChange, onSelect, onCursor, getClass, classInfo }) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.onChange = onChange;   // (anns, {commit}) after an edit
    this.onSelect = onSelect;   // (index)
    this.onCursor = onCursor;   // (x, y) in image px or null
    this.getClass = getClass;   // () -> current class id
    this.classInfo = classInfo; // (id) -> {name, color}
    this.img = null;
    this.anns = [];
    this.sel = -1;
    this.view = { s: 1, ox: 0, oy: 0 };
    this.showBoxes = true;
    this.showTags = true;
    this.drag = null;
    this.space = false;
    this._bind();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
  }

  // ------------------------------------------------------------ public
  setImage(img, keepView = false) {
    const same = this.img && img && this.img.naturalWidth === img.naturalWidth &&
      this.img.naturalHeight === img.naturalHeight;
    this.img = img;
    if (!(keepView && same)) this.fit();
    this.draw();
  }
  setAnnotations(anns) { this.anns = anns; if (this.sel >= anns.length) this.sel = -1; this.draw(); }
  select(i) { this.sel = i; this.onSelect?.(i); this.draw(); }
  get width() { return this.img ? this.img.naturalWidth : 0; }
  get height() { return this.img ? this.img.naturalHeight : 0; }

  fit() {
    if (!this.img) return;
    const W = this.cv.clientWidth, H = this.cv.clientHeight, pad = 16;
    const s = Math.min((W - 2 * pad) / this.width, (H - 2 * pad) / this.height);
    this.view = { s, ox: (W - this.width * s) / 2, oy: (H - this.height * s) / 2 };
    this.draw();
  }

  nudge(dx, dy) {
    const a = this.anns[this.sel];
    if (!a) return;
    const w = a.x2 - a.x1, h = a.y2 - a.y1;
    a.x1 = clamp(a.x1 + dx, 0, this.width - w); a.y1 = clamp(a.y1 + dy, 0, this.height - h);
    a.x2 = a.x1 + w; a.y2 = a.y1 + h;
    touch(a);
    this.onChange?.(this.anns, { commit: true });
    this.draw();
  }

  // ------------------------------------------------------------ geometry
  toImg(px, py) { return [(px - this.view.ox) / this.view.s, (py - this.view.oy) / this.view.s]; }
  toScr(x, y) { return [x * this.view.s + this.view.ox, y * this.view.s + this.view.oy]; }

  _hitHandle(px, py) {
    const a = this.anns[this.sel];
    if (!a) return null;
    const [x1, y1] = this.toScr(a.x1, a.y1), [x2, y2] = this.toScr(a.x2, a.y2);
    const xm = (x1 + x2) / 2, ym = (y1 + y2) / 2, r = HANDLE + 2;
    const hs = { nw: [x1, y1], ne: [x2, y1], sw: [x1, y2], se: [x2, y2],
                 n: [xm, y1], s: [xm, y2], w: [x1, ym], e: [x2, ym] };
    for (const [k, [hx, hy]] of Object.entries(hs))
      if (Math.abs(px - hx) <= r && Math.abs(py - hy) <= r) return k;
    return null;
  }

  _hitBox(px, py) {
    const [x, y] = this.toImg(px, py);
    const tol = 3 / this.view.s;
    const inside = (a) => x >= a.x1 - tol && x <= a.x2 + tol && y >= a.y1 - tol && y <= a.y2 + tol;
    if (this.sel >= 0 && this.anns[this.sel] && inside(this.anns[this.sel])) return this.sel;
    // smallest containing box wins (easier to pick nested/overlapping boxes)
    let best = -1, area = Infinity;
    this.anns.forEach((a, i) => {
      if (inside(a)) { const ar = (a.x2 - a.x1) * (a.y2 - a.y1); if (ar < area) { area = ar; best = i; } }
    });
    return best;
  }

  // ------------------------------------------------------------ events
  _bind() {
    const cv = this.cv;
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("wheel", (e) => {
      if (!this.img) return;
      e.preventDefault();
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const f = Math.exp(-e.deltaY * 0.0015);
      const s = clamp(this.view.s * f, 0.05, 80);
      this.view.ox = px - (px - this.view.ox) * (s / this.view.s);
      this.view.oy = py - (py - this.view.oy) * (s / this.view.s);
      this.view.s = s;
      this.draw();
    }, { passive: false });

    cv.addEventListener("pointerdown", (e) => {
      if (!this.img) return;
      cv.setPointerCapture(e.pointerId);
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      if (e.button === 1 || e.button === 2 || (e.button === 0 && this.space)) {
        this.drag = { kind: "pan", px, py, ox: this.view.ox, oy: this.view.oy };
        cv.style.cursor = "grabbing";
        return;
      }
      if (e.button !== 0) return;
      const [x, y] = this.toImg(px, py);
      const h = this.showBoxes ? this._hitHandle(px, py) : null;
      if (h) {
        this.drag = { kind: "resize", h, orig: { ...this.anns[this.sel] } };
        return;
      }
      const hit = this.showBoxes ? this._hitBox(px, py) : -1;
      if (hit >= 0) {
        this.select(hit);
        this.drag = { kind: "move", x, y, orig: { ...this.anns[hit] }, moved: false };
        return;
      }
      if (x < 0 || y < 0 || x > this.width || y > this.height) { this.select(-1); return; }
      const a = { type: "bbox", cls: this.getClass(), x1: x, y1: y, x2: x, y2: y, conf: null, source: "manual" };
      this.anns.push(a);
      this.sel = this.anns.length - 1;
      this.drag = { kind: "new", x, y };
      this.showBoxes = true;
      this.draw();
    });

    cv.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      if (!this.img) return;
      const [x, y] = this.toImg(px, py);
      this.onCursor?.(x >= 0 && y >= 0 && x < this.width && y < this.height ? [x, y] : null);
      const d = this.drag;
      if (!d) { this._hoverCursor(px, py); return; }
      const W = this.width, H = this.height;
      const cx = clamp(x, 0, W), cy = clamp(y, 0, H);
      if (d.kind === "pan") {
        this.view.ox = d.ox + (px - d.px); this.view.oy = d.oy + (py - d.py);
      } else if (d.kind === "new") {
        const a = this.anns[this.sel];
        a.x1 = Math.min(d.x, cx); a.x2 = Math.max(d.x, cx);
        a.y1 = Math.min(d.y, cy); a.y2 = Math.max(d.y, cy);
      } else if (d.kind === "move") {
        const a = this.anns[this.sel], o = d.orig, w = o.x2 - o.x1, h = o.y2 - o.y1;
        a.x1 = clamp(o.x1 + (x - d.x), 0, W - w); a.y1 = clamp(o.y1 + (y - d.y), 0, H - h);
        a.x2 = a.x1 + w; a.y2 = a.y1 + h;
        d.moved = d.moved || Math.abs(x - d.x) * this.view.s > 1 || Math.abs(y - d.y) * this.view.s > 1;
      } else if (d.kind === "resize") {
        const a = this.anns[this.sel], o = d.orig;
        let { x1, y1, x2, y2 } = o;
        if (d.h.includes("w")) x1 = cx; if (d.h.includes("e")) x2 = cx;
        if (d.h.includes("n")) y1 = cy; if (d.h.includes("s")) y2 = cy;
        a.x1 = Math.min(x1, x2); a.x2 = Math.max(x1, x2);
        a.y1 = Math.min(y1, y2); a.y2 = Math.max(y1, y2);
      }
      this.draw();
    });

    const end = () => {
      const d = this.drag;
      this.drag = null;
      this.cv.style.cursor = "";
      if (!d || d.kind === "pan") return;
      const a = this.anns[this.sel];
      if (d.kind === "new") {
        if (!a || a.x2 - a.x1 < MIN_BOX || a.y2 - a.y1 < MIN_BOX) {
          this.anns.splice(this.sel, 1); this.select(-1); return;
        }
        this.onSelect?.(this.sel);
        this.onChange?.(this.anns, { commit: true });
      } else if (d.kind === "resize" || (d.kind === "move" && d.moved)) {
        touch(a);
        this.onChange?.(this.anns, { commit: true });
      }
      this.draw();
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.addEventListener("pointerleave", () => this.onCursor?.(null));

    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !isTyping(e)) { this.space = true; if (!this.drag) this.cv.style.cursor = "grab"; e.preventDefault(); }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") { this.space = false; if (!this.drag) this.cv.style.cursor = ""; }
    });
  }

  _hoverCursor(px, py) {
    if (this.space) { this.cv.style.cursor = "grab"; return; }
    const h = this.showBoxes ? this._hitHandle(px, py) : null;
    const map = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
                  n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };
    if (h) this.cv.style.cursor = map[h];
    else if (this.showBoxes && this._hitBox(px, py) >= 0) this.cv.style.cursor = "move";
    else this.cv.style.cursor = "crosshair";
  }

  // ------------------------------------------------------------ drawing
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const p = this.cv.parentElement;
    this.cv.width = Math.max(1, Math.round(p.clientWidth * dpr));
    this.cv.height = Math.max(1, Math.round(p.clientHeight * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.img && !this._fitted) { this._fitted = true; this.fit(); }
    this.draw();
  }

  draw() {
    const c = this.ctx, W = this.cv.clientWidth, H = this.cv.clientHeight;
    c.clearRect(0, 0, W, H);
    if (!this.img) return;
    const { s, ox, oy } = this.view;
    c.imageSmoothingEnabled = s < 1;
    c.drawImage(this.img, ox, oy, this.width * s, this.height * s);
    c.strokeStyle = "#444"; c.lineWidth = 1;
    c.strokeRect(ox - 0.5, oy - 0.5, this.width * s + 1, this.height * s + 1);
    if (!this.showBoxes) return;
    c.font = "11px system-ui, sans-serif";
    c.textBaseline = "bottom";
    this.anns.forEach((a, i) => {
      if (i !== this.sel) this._drawBox(a, false);
    });
    if (this.anns[this.sel]) this._drawBox(this.anns[this.sel], true);
  }

  _drawBox(a, selected) {
    const c = this.ctx, info = this.classInfo(a.cls);
    const [x1, y1] = this.toScr(a.x1, a.y1), [x2, y2] = this.toScr(a.x2, a.y2);
    c.save();
    c.strokeStyle = info.color;
    c.lineWidth = selected ? 2.5 : 1.5;
    if (a.source === "model") c.setLineDash([5, 3]);
    c.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (selected) {
      c.fillStyle = hexA(info.color, 0.12);
      c.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    c.setLineDash([]);
    if (this.showTags && (selected || (x2 - x1) > 18)) {
      const t = a.conf != null ? `${info.name} ${a.conf.toFixed(2)}` : info.name;
      const tw = c.measureText(t).width + 6;
      const ty = y1 > 14 ? y1 : y2 + 14;
      c.fillStyle = hexA(info.color, 0.85);
      c.fillRect(x1 - (selected ? 1.25 : 0.75), ty - 14, tw, 14);
      c.fillStyle = textOn(info.color);
      c.fillText(t, x1 + 2, ty - 1);
    }
    if (selected) {
      const xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
      c.fillStyle = "#fff"; c.strokeStyle = "#000"; c.lineWidth = 1;
      for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2], [xm, y1], [xm, y2], [x1, ym], [x2, ym]]) {
        c.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
        c.strokeRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
      }
    }
    c.restore();
  }
}

// a model box that is edited becomes "edited"; others keep their source
export function touch(a) { if (a && a.source === "model") a.source = "edited"; }

export function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" && !["checkbox", "range", "radio"].includes(t.type) ||
    t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }
function hexA(hex, a) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex || "");
  if (!m) return `rgba(200,200,200,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}
function textOn(hex) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex || "");
  if (!m) return "#000";
  const l = 0.299 * parseInt(m[1], 16) + 0.587 * parseInt(m[2], 16) + 0.114 * parseInt(m[3], 16);
  return l > 140 ? "#000" : "#fff";
}
