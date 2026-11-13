// Movies in the page: an MP4/MOV demuxer and WebCodecs, serving decoded
// frames to wobble by index (webmain.rs, `WebMovies`).
//
// wobble asks for a frame synchronously and cannot wait, so this side works
// ahead: a request for a frame it does not have queues the decode of that
// frame's whole GOP (keyframe to keyframe) and returns nothing; the player
// shows black for that tick and asks again, and when the GOP lands we tell
// wobble (`wobble_movie_ready`) so a paused picture refreshes. Frames are
// indexed in PRESENTATION order (by timestamp), the rule the desktop
// decoder follows. Pixels are code values (RGBA8); the Read's input space
// converts them, as for any other file.

const movies = new Map(); // path -> Movie

// ------------------------------------------------------------------ demux

function boxes(view, start, end) {
  const out = [];
  let p = start;
  while (p + 8 <= end) {
    let size = view.getUint32(p);
    const type = String.fromCharCode(view.getUint8(p + 4), view.getUint8(p + 5), view.getUint8(p + 6), view.getUint8(p + 7));
    let head = 8;
    if (size === 1) {
      size = Number(view.getBigUint64(p + 8));
      head = 16;
    } else if (size === 0) {
      size = end - p;
    }
    if (size < head || p + size > end) break;
    out.push({ type, start: p, body: p + head, end: p + size });
    p += size;
  }
  return out;
}

const child = (view, box, type) => boxes(view, box.body, box.end).find((b) => b.type === type);
const path = (view, box, ...types) => types.reduce((b, t) => b && child(view, b, t), box);

// The top-level boxes, read header by header so a large file is never read
// whole: `moov` is all the demuxer needs; samples are sliced per GOP.
async function findMoov(file) {
  let p = 0;
  while (p + 8 <= file.size) {
    const h = new DataView(await file.slice(p, p + 16).arrayBuffer());
    let size = h.getUint32(0);
    const type = String.fromCharCode(h.getUint8(4), h.getUint8(5), h.getUint8(6), h.getUint8(7));
    if (size === 1) size = Number(h.getBigUint64(8));
    else if (size === 0) size = file.size - p;
    if (size < 8) break;
    if (type === "moov") return new DataView(await file.slice(p, p + size).arrayBuffer());
    p += size;
  }
  throw new Error("no movie header (moov) in this file");
}

const hex = (n) => n.toString(16).padStart(2, "0");

function codecOf(view, entry) {
  const t = entry.type;
  const kids = boxes(view, entry.body + 78, entry.end);
  const bytes = (b) => new Uint8Array(view.buffer, view.byteOffset + b.body, b.end - b.body);
  if (t === "avc1" || t === "avc3") {
    const c = kids.find((b) => b.type === "avcC");
    if (!c) throw new Error("H.264 track without its avcC");
    const d = bytes(c);
    return { codec: `avc1.${hex(d[1])}${hex(d[2])}${hex(d[3])}`, description: d.slice() };
  }
  if (t === "hvc1" || t === "hev1") {
    const c = kids.find((b) => b.type === "hvcC");
    if (!c) throw new Error("HEVC track without its hvcC");
    const d = bytes(c);
    const space = ["", "A", "B", "C"][d[1] >> 6];
    const tier = (d[1] >> 5) & 1 ? "H" : "L";
    const profile = d[1] & 31;
    let compat = ((d[2] << 24) | (d[3] << 16) | (d[4] << 8) | d[5]) >>> 0;
    let rev = 0;
    for (let i = 0; i < 32; i++) rev = ((rev << 1) | ((compat >>> i) & 1)) >>> 0;
    const cons = [...d.slice(6, 12)];
    while (cons.length && cons[cons.length - 1] === 0) cons.pop();
    const tail = cons.length ? "." + cons.map((b) => b.toString(16)).join(".") : "";
    return { codec: `${t}.${space}${profile}.${rev.toString(16)}.${tier}${d[12]}${tail}`, description: d.slice() };
  }
  if (t === "av01") {
    const c = kids.find((b) => b.type === "av1C");
    if (!c) throw new Error("AV1 track without its av1C");
    const d = bytes(c);
    const profile = d[1] >> 5, level = d[1] & 31, tierH = d[2] >> 7;
    const depth = (d[2] >> 6) & 1 ? ((d[2] >> 5) & 1 ? 12 : 10) : 8;
    return { codec: `av01.${profile}.${String(level).padStart(2, "0")}${tierH ? "H" : "M"}.${String(depth).padStart(2, "0")}`, description: d.slice() };
  }
  if (t === "vp09") {
    const c = kids.find((b) => b.type === "vpcC");
    if (!c) throw new Error("VP9 track without its vpcC");
    const d = bytes(c); // full box: 4 bytes version/flags first
    return { codec: `vp09.${hex(d[4])}.${hex(d[5])}.${hex(d[6] >> 4)}` };
  }
  const prores = ["apcn", "apch", "apcs", "apco", "ap4h", "ap4x"].includes(t);
  throw new Error(prores
    ? "this is ProRes, which no browser can decode; the desktop app plays it"
    : `video in '${t}', which this page does not decode (H.264, HEVC, AV1 and VP9 do)`);
}

function demux(view) {
  const moov = { body: 8, end: view.byteLength };
  for (const trak of boxes(view, moov.body, moov.end).filter((b) => b.type === "trak")) {
    const mdia = child(view, trak, "mdia");
    const hdlr = mdia && child(view, mdia, "hdlr");
    if (!hdlr) continue;
    const kind = String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + hdlr.body + 8, 4));
    if (kind !== "vide") continue;
    const mdhd = child(view, mdia, "mdhd");
    const timescale = view.getUint8(mdhd.body) === 1 ? view.getUint32(mdhd.body + 20) : view.getUint32(mdhd.body + 12);
    const stbl = path(view, mdia, "minf", "stbl");
    const stsd = child(view, stbl, "stsd");
    const entry = boxes(view, stsd.body + 8, stsd.end)[0];
    const width = view.getUint16(entry.body + 24), height = view.getUint16(entry.body + 26);
    const { codec, description } = codecOf(view, entry);

    const sizes = [];
    const stsz = child(view, stbl, "stsz");
    const fixed = view.getUint32(stsz.body + 4), count = view.getUint32(stsz.body + 8);
    for (let i = 0; i < count; i++) sizes.push(fixed || view.getUint32(stsz.body + 12 + i * 4));

    const chunks = [];
    const stco = child(view, stbl, "stco"), co64 = child(view, stbl, "co64");
    if (stco) for (let i = 0, n = view.getUint32(stco.body + 4); i < n; i++) chunks.push(view.getUint32(stco.body + 8 + i * 4));
    if (co64) for (let i = 0, n = view.getUint32(co64.body + 4); i < n; i++) chunks.push(Number(view.getBigUint64(co64.body + 8 + i * 8)));

    // Sample offsets: walk the chunks, each holding stsc's samples-per-chunk.
    const stsc = child(view, stbl, "stsc");
    const runs = [];
    for (let i = 0, n = view.getUint32(stsc.body + 4); i < n; i++) {
      const o = stsc.body + 8 + i * 12;
      runs.push([view.getUint32(o) - 1, view.getUint32(o + 4)]);
    }
    const offsets = new Array(count);
    let s = 0;
    for (let c = 0; c < chunks.length && s < count; c++) {
      let per = 0;
      for (const [first, n] of runs) if (first <= c) per = n;
      let off = chunks[c];
      for (let k = 0; k < per && s < count; k++, s++) {
        offsets[s] = off;
        off += sizes[s];
      }
    }

    const dts = new Array(count);
    const stts = child(view, stbl, "stts");
    let t = 0;
    s = 0;
    for (let i = 0, n = view.getUint32(stts.body + 4); i < n; i++) {
      const c = view.getUint32(stts.body + 8 + i * 8), d = view.getUint32(stts.body + 12 + i * 8);
      for (let k = 0; k < c && s < count; k++, s++) { dts[s] = t; t += d; }
    }
    const pts = dts.slice();
    const ctts = child(view, stbl, "ctts");
    if (ctts) {
      const v1 = view.getUint8(ctts.body) === 1;
      s = 0;
      for (let i = 0, n = view.getUint32(ctts.body + 4); i < n; i++) {
        const c = view.getUint32(ctts.body + 8 + i * 8);
        const o = v1 ? view.getInt32(ctts.body + 12 + i * 8) : view.getUint32(ctts.body + 12 + i * 8);
        for (let k = 0; k < c && s < count; k++, s++) pts[s] += o;
      }
    }
    const sync = new Array(count).fill(false);
    const stss = child(view, stbl, "stss");
    if (stss) for (let i = 0, n = view.getUint32(stss.body + 4); i < n; i++) sync[view.getUint32(stss.body + 8 + i * 4) - 1] = true;
    else sync.fill(true);

    // Presentation order: frame i is sample order[i].
    const order = [...Array(count).keys()].sort((a, b) => pts[a] - pts[b]);
    const indexOf = new Map(order.map((smp, i) => [Math.round((pts[smp] * 1e6) / timescale), i]));
    return { width, height, codec, description, timescale, sizes, offsets, pts, sync, order, indexOf };
  }
  throw new Error("no video track in this file");
}

// ---------------------------------------------------------------- decode

class Movie {
  constructor(file, d) {
    Object.assign(this, d);
    this.file = file;
    this.frames = new Map(); // presentation index -> Uint8ClampedArray (RGBA)
    this.queue = [];         // GOP starts (decode order) waiting
    this.busy = false;
    this.missed = false;
    this.last = 0;
    this.budget = Math.max(12, Math.floor((768 << 20) / (this.width * this.height * 4)));
    this.canvas = new OffscreenCanvas(this.width, this.height);
    this.ctx2d = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  gopOf(i) {
    let k = this.order[i];
    while (k > 0 && !this.sync[k]) k--;
    return k;
  }

  frame(i) {
    this.last = i;
    const px = this.frames.get(i);
    // Keep a GOP ahead of the playhead.
    const ahead = Math.min(this.order.length - 1, i + 12);
    if (!this.frames.has(ahead)) this.want(ahead);
    if (px) return px;
    this.missed = true;
    this.want(i);
    return undefined;
  }

  want(i) {
    const k = this.gopOf(i);
    if (!this.queue.includes(k) && this.current !== k) this.queue.unshift(k);
    this.pump();
  }

  async pump() {
    if (this.busy || !this.queue.length) return;
    this.busy = true;
    const k = this.queue.shift();
    this.current = k;
    try {
      await this.decodeGop(k);
    } catch (e) {
      console.warn("movie decode:", e);
    }
    this.current = undefined;
    this.busy = false;
    if (this.missed) {
      this.missed = false;
      window.wobble && window.wobble.wobble_movie_ready();
    }
    this.pump();
  }

  async decodeGop(k) {
    let end = k + 1;
    while (end < this.sync.length && !this.sync[end]) end++;
    const lo = this.offsets[k];
    const hi = Math.max(...this.offsets.slice(k, end).map((o, j) => o + this.sizes[k + j]));
    const bytes = new Uint8Array(await this.file.slice(lo, hi).arrayBuffer());
    const keep = (i) => Math.abs(i - this.last) <= this.budget / 2;
    await new Promise((resolve, reject) => {
      const dec = new VideoDecoder({
        output: (vf) => {
          const i = this.indexOf.get(vf.timestamp);
          if (i !== undefined && keep(i) && !this.frames.has(i)) {
            this.ctx2d.drawImage(vf, 0, 0, this.width, this.height);
            this.frames.set(i, this.ctx2d.getImageData(0, 0, this.width, this.height).data);
          }
          vf.close();
        },
        error: reject,
      });
      dec.configure({ codec: this.codec, description: this.description, codedWidth: this.width, codedHeight: this.height, hardwareAcceleration: "prefer-hardware" });
      for (let s = k; s < end; s++) {
        const o = this.offsets[s] - lo;
        dec.decode(new EncodedVideoChunk({
          type: this.sync[s] ? "key" : "delta",
          timestamp: Math.round((this.pts[s] * 1e6) / this.timescale),
          data: bytes.subarray(o, o + this.sizes[s]),
        }));
      }
      dec.flush().then(() => { dec.close(); resolve(); }, reject);
    });
    // Over budget: drop what is farthest from the playhead.
    if (this.frames.size > this.budget) {
      const far = [...this.frames.keys()].sort((a, b) => Math.abs(b - this.last) - Math.abs(a - this.last));
      for (const i of far.slice(0, this.frames.size - this.budget)) this.frames.delete(i);
    }
  }
}

export const isMovie = (name) => /\.(mp4|mov|m4v)$/i.test(name);

/// Demux `file` and check this browser decodes it; throws a message a
/// person can read when it cannot.
export async function addMovie(p, file) {
  if (typeof VideoDecoder === "undefined") throw new Error("this browser has no WebCodecs, so it cannot play movies");
  const d = demux(await findMoov(file));
  const ok = await VideoDecoder.isConfigSupported({ codec: d.codec, description: d.description, codedWidth: d.width, codedHeight: d.height });
  if (!ok.supported) throw new Error(`this browser can't decode ${d.codec}`);
  movies.set(p, new Movie(file, d));
}

window.wobbleMovieInfo = (p) => {
  const m = movies.get(p);
  return m ? new Uint32Array([m.width, m.height, m.order.length]) : undefined;
};

window.wobbleMovieFrame = (p, i) => {
  const m = movies.get(p);
  if (!m) return undefined;
  const px = m.frame(i);
  return px ? new Uint8Array(px.buffer, px.byteOffset, px.byteLength) : undefined;
};
