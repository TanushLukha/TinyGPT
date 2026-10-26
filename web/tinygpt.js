// TinyGPT in the browser (M6): the tokenizer, the model's forward pass, the samplers and the
// parser, rewritten in plain JavaScript from the Python code in src/tinygpt. No libraries, no
// server: a page downloads the two bundles (scripts/export.py) and runs everything on the
// visitor's own processor. tests/test_web.py checks every piece against the Python version.
//
//   normalize          src/tinygpt/text.py      NFC, control characters, spaces
//   Tokenizer          src/tinygpt/bpe.py       byte-level BPE with the Devanagari-aware regex
//   readSafetensors    src/tinygpt/bundle.py    the weights file: a JSON header + raw float32
//   GPT                src/tinygpt/models/gpt.py   Llama recipe: RoPE, RMSNorm, SwiGLU, KV-cache
//   filterLogits       src/tinygpt/sampling.py  repetition penalty, temperature, top-k/p, min-p
//   stream             src/tinygpt/serve/engine.py  TextGenerator.stream
//   Parser, parseTree  src/tinygpt/top.py, serve/engine.py

// ------------------------------------------------------------------ text (text.py)
// Python's str.isspace(): what str.strip() removes
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");
export const pyStrip = (s) => s.replace(STRIP, "");

/** NFC, \r\n -> \n, control/format characters removed (except \n, ZWNJ, ZWJ), runs of spaces -> one,
 *  every line stripped, at most one blank line in a row. Same as text.normalize. */
export function normalize(text) {
  text = text.normalize("NFC").replace(/\r\n?/g, "\n");
  text = text.replace(/\p{C}/gu, (ch) => (ch === "\n" || ch === "\u200c" || ch === "\u200d" ? ch : ""));
  text = text.replace(/[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+/g, " ");
  text = text.split("\n").map(pyStrip).join("\n");
  return pyStrip(text.replace(/\n{3,}/g, "\n\n"));
}

// ------------------------------------------------------------------ tokenizer (bpe.py)
// The "indic" pre-tokenisation pattern, written for JavaScript's regex engine:
//  * (?i:[sdmt]|ll|ve|re) -> spelled-out case variants (ſ folds to s in Python's matcher);
//  * possessive ?+ and ++ -> plain ? and +: here nothing after them could match what they
//    give back, so backtracking never changes a match;
//  * \s -> the exact 25 code points Python's `regex` module calls whitespace (JavaScript's \s
//    adds U+FEFF and lacks U+0085).
const RX_SPACE = "\\t\\n\\v\\f\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const W = "\\p{L}\\p{M}\\u200c\\u200d";
const INDIC = new RegExp(
  "'(?:[sdmtSDMTſ]|[lL][lL]|[vV][eE]|[rR][eE])" +
  `|[^\\r\\n${W}\\p{N}]?[${W}]+` +
  "|\\p{N}{1,3}" +
  `| ?[^${RX_SPACE}${W}\\p{N}]+[\\r\\n]*` +
  `|[${RX_SPACE}]*[\\r\\n]` +
  `|[${RX_SPACE}]+(?![^${RX_SPACE}])` +
  `|[${RX_SPACE}]+`,
  "gu");

const utf8 = new TextEncoder();

export class Tokenizer {
  /** `data`: the parsed tokenizer.json (type "tinygpt-bpe"). */
  constructor(data) {
    if (data.type !== "tinygpt-bpe" || data.pattern_name !== "indic") {
      throw new Error("expected a tinygpt-bpe tokenizer with the indic pattern");
    }
    this.merges = data.merges;
    this.ranks = new Map(data.merges.map(([a, b], r) => [a * 65536 + b, r]));   // pair -> rank
    this.vocab = [];                                                              // id -> bytes
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    for (const [a, b] of data.merges) {
      const x = this.vocab[a], y = this.vocab[b], m = new Uint8Array(x.length + y.length);
      m.set(x); m.set(y, x.length);
      this.vocab.push(m);
    }
    this.special = new Map(data.special_tokens.map((s, i) => [s, 256 + data.merges.length + i]));
    for (const [s] of this.special) this.vocab.push(utf8.encode(s));
    this.eotId = this.special.get("<|endoftext|>");
    this.vocabSize = this.vocab.length;
    this.cache = new Map();
  }

  encodeChunk(chunk) {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    let ids = Array.from(utf8.encode(chunk));
    while (ids.length >= 2) {          // merge the pair learned earliest, everywhere, until none is left
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = ids[best], b = ids[best + 1], merged = [];
      for (let i = 0; i < ids.length;) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { merged.push(256 + bestRank); i += 2; }
        else { merged.push(ids[i]); i += 1; }
      }
      ids = merged;
    }
    if (this.cache.size > 100000) this.cache.clear();
    this.cache.set(chunk, ids);
    return ids;
  }

  /** Text -> token ids; special-token strings are treated as plain text (encode_ordinary). */
  encode(text) {
    const out = [];
    for (const m of text.matchAll(INDIC)) for (const id of this.encodeChunk(m[0])) out.push(id);
    return out;
  }

  decodeBytes(ids) {
    let n = 0;
    for (const i of ids) n += this.vocab[i].length;
    const out = new Uint8Array(n);
    let o = 0;
    for (const i of ids) { out.set(this.vocab[i], o); o += this.vocab[i].length; }
    return out;
  }

  decode(ids) { return new TextDecoder().decode(this.decodeBytes(ids)); }
}

// ------------------------------------------------------------------ weights (bundle.py)
/** A .safetensors file: 8 bytes of header length, a JSON header, then the raw tensors. */
export function readSafetensors(buffer) {
  const n = Number(new DataView(buffer).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
  const start = 8 + n, tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`${name}: ${info.dtype}, expected F32`);
    const [a, b] = info.data_offsets, at = start + a, count = (b - a) / 4;
    tensors[name] = (at % 4 === 0) ? new Float32Array(buffer, at, count)
                                   : new Float32Array(buffer.slice(at, start + b));   // unaligned: copy
  }
  // a tied matrix is stored once; the metadata names the copy that was dropped
  for (const [dropped, kept] of Object.entries(header.__metadata__ || {})) {
    if (tensors[kept] && !tensors[dropped]) tensors[dropped] = tensors[kept];
  }
  return tensors;
}

// ------------------------------------------------------------------ the model (models/gpt.py)
/** out[r] = W[r, :] · x for a row-major (rows × cols) matrix: one output per row, as nn.Linear. */
function matvec(W, x, out, rows, cols) {
  for (let r = 0, o = 0; r < rows; r++, o += cols) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
    for (; c + 3 < cols; c += 4) {
      s0 += W[o + c] * x[c]; s1 += W[o + c + 1] * x[c + 1];
      s2 += W[o + c + 2] * x[c + 2]; s3 += W[o + c + 3] * x[c + 3];
    }
    for (; c < cols; c++) s0 += W[o + c] * x[c];
    out[r] = (s0 + s1) + (s2 + s3);
  }
}

function rmsnorm(x, weight, out, eps = 1e-6) {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * weight[i];
}

export class GPT {
  /** `config`: model_config from the bundle's config.json; `tensors`: readSafetensors(...). */
  constructor(config, tensors) {
    if (config.pos !== "rope" || config.norm !== "rmsnorm" || config.mlp !== "swiglu" || !config.tie_weights) {
      throw new Error("the browser version runs the Llama recipe (RoPE, RMSNorm, SwiGLU, tied weights)");
    }
    const d = this.d = config.d_model;
    this.H = config.n_heads; this.hd = d / this.H; this.L = config.n_layers;
    this.V = config.vocab_size; this.T = config.block_size;
    const t = (name) => { if (!tensors[name]) throw new Error(`missing ${name}`); return tensors[name]; };
    this.emb = t("head.weight");                                   // (V, d): embedding = output layer
    this.blocks = [];
    for (let i = 0; i < this.L; i++) {
      const p = `blocks.${i}.`;
      this.blocks.push({ ln1: t(p + "ln1.weight"), qkv: t(p + "attn.qkv.weight"), proj: t(p + "attn.proj.weight"),
                         ln2: t(p + "ln2.weight"), gate: t(p + "mlp.gate.weight"), up: t(p + "mlp.up.weight"),
                         down: t(p + "mlp.proj.weight") });
    }
    this.hidden = this.blocks[0].gate.length / d;
    this.lnf = t("ln_f.weight");
    // RoPE angles in float32, as rope_tables computes them: position p turns pair i by p * base^(-2i/hd)
    const half = this.hd / 2, f = Math.fround;
    this.cos = new Float32Array(this.T * half); this.sin = new Float32Array(this.T * half);
    for (let i = 0; i < half; i++) {
      const inv = f(1 / f(Math.pow(config.rope_base, f(f(2 * i) / this.hd))));
      for (let p = 0; p < this.T; p++) {
        const a = f(p * inv);
        this.cos[p * half + i] = Math.cos(a); this.sin[p * half + i] = Math.sin(a);
      }
    }
    this.x = new Float32Array(d); this.h = new Float32Array(d); this.qkvOut = new Float32Array(3 * d);
    this.att = new Float32Array(d); this.tmp = new Float32Array(d);
    this.g = new Float32Array(this.hidden); this.u = new Float32Array(this.hidden);
    this.scores = new Float64Array(this.T);
    this.logits = new Float32Array(this.V);
  }

  newCache() {
    const size = this.T * this.d;
    return { pos: 0, k: this.blocks.map(() => new Float32Array(size)), v: this.blocks.map(() => new Float32Array(size)) };
  }

  rope(vec, offset, pos) {                      // rotate pairs (0,1), (2,3), ... of one head in place
    const half = this.hd / 2, base = pos * half;
    for (let i = 0; i < half; i++) {
      const c = this.cos[base + i], s = this.sin[base + i];
      const a = vec[offset + 2 * i], b = vec[offset + 2 * i + 1];
      vec[offset + 2 * i] = a * c - b * s;
      vec[offset + 2 * i + 1] = a * s + b * c;
    }
  }

  /** Feed one token at the next position; returns the logits for the token after it (or null). */
  step(token, cache, wantLogits = true) {
    const { d, H, hd, x, h, qkvOut, att, tmp, g, u, scores } = this;
    const pos = cache.pos;
    if (pos >= this.T) throw new Error("the context window is full");
    x.set(this.emb.subarray(token * d, token * d + d));
    const scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < this.L; l++) {
      const B = this.blocks[l], K = cache.k[l], Vc = cache.v[l];
      rmsnorm(x, B.ln1, h);
      matvec(B.qkv, h, qkvOut, 3 * d, d);                      // queries, keys, values at once
      for (let hh = 0; hh < H; hh++) {
        this.rope(qkvOut, hh * hd, pos);                       // q
        this.rope(qkvOut, d + hh * hd, pos);                   // k
      }
      K.set(qkvOut.subarray(d, 2 * d), pos * d);               // the cache keeps this token's k, v
      Vc.set(qkvOut.subarray(2 * d, 3 * d), pos * d);
      for (let hh = 0; hh < H; hh++) {                         // each head attends to positions 0..pos
        const qo = hh * hd;
        let max = -Infinity;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          const ko = j * d + qo;
          for (let c = 0; c < hd; c++) s += qkvOut[qo + c] * K[ko + c];
          scores[j] = s * scale;
          if (scores[j] > max) max = scores[j];
        }
        let sum = 0;
        for (let j = 0; j <= pos; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }
        for (let c = 0; c < hd; c++) {
          let s = 0;
          for (let j = 0; j <= pos; j++) s += scores[j] * Vc[j * d + qo + c];
          att[qo + c] = s / sum;
        }
      }
      matvec(B.proj, att, tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
      rmsnorm(x, B.ln2, h);                                    // SwiGLU: silu(gate(h)) * up(h), then down
      matvec(B.gate, h, g, this.hidden, d);
      matvec(B.up, h, u, this.hidden, d);
      for (let i = 0; i < this.hidden; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      matvec(B.down, g, tmp, d, this.hidden);
      for (let i = 0; i < d; i++) x[i] += tmp[i];
    }
    cache.pos = pos + 1;
    if (!wantLogits) return null;
    rmsnorm(x, this.lnf, h);
    matvec(this.emb, h, this.logits, this.V, d);
    return this.logits;
  }
}

// ------------------------------------------------------------------ sampling (sampling.py)
/** A seeded generator of numbers in [0, 1) (mulberry32): the same seed, the same text. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const p = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - max); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

/** Every strategy except the final draw, in sampling.py's order: repetition penalty, temperature,
 *  top-k, top-p, min-p. Removed tokens get -Infinity. `history`: the token ids so far. */
export function filterLogits(logits, { temperature = 1, topK = null, topP = null, minP = null,
                                       repetitionPenalty = 1, history = [] } = {}) {
  const out = Float32Array.from(logits);
  if (repetitionPenalty !== 1 && history.length) {
    for (const id of new Set(history)) {
      const v = logits[id];
      out[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= temperature;
  if (topK && topK < out.length) {
    const kth = Float32Array.from(out).sort()[out.length - topK];
    for (let i = 0; i < out.length; i++) if (out[i] < kth) out[i] = -Infinity;
  }
  if (topP !== null && topP < 1) {
    const p = softmax(out), order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
    let before = 0;
    for (const i of order) { if (before > topP) out[i] = -Infinity; before += p[i]; }
  }
  if (minP) {
    const p = softmax(out);
    let max = 0;
    for (const v of p) if (v > max) max = v;
    for (let i = 0; i < out.length; i++) if (p[i] < minP * max) out[i] = -Infinity;
  }
  return out;
}

function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

/** The next token: greedy at temperature 0 (the filters can't change the argmax), else a draw. */
export function sampleNext(logits, params, random) {
  if (params.temperature === 0) {
    return argmax(filterLogits(logits, { repetitionPenalty: params.repetitionPenalty, history: params.history }));
  }
  const p = softmax(filterLogits(logits, params));
  let u = random(), last = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) { last = i; u -= p[i]; if (u < 0) return i; }
  }
  return last;                                   // rounding left a sliver: the last possible token
}

/** Continue `ids`, yielding each new token (models/gpt.py iter_tokens, with its window restart:
 *  when the window is full, the cache restarts from the last half of the text). */
export function* iterTokens(model, ids, maxNewTokens, params, random) {
  const history = ids.slice();
  let cache = model.newCache(), feed = ids.slice(-model.T);
  for (let n = 0; n < maxNewTokens; n++) {
    let logits = null;
    for (let i = 0; i < feed.length; i++) logits = model.step(feed[i], cache, i === feed.length - 1);
    const next = sampleNext(logits, { ...params, history }, random);
    history.push(next);
    yield next;
    if (params.eotId !== undefined && next === params.eotId) return;
    if (cache.pos < model.T) feed = [next];
    else { cache = model.newCache(); feed = history.slice(-(model.T >> 1)); }
  }
}

/** Generation as the server streams it (engine.TextGenerator.stream): {text} pieces of whole
 *  characters, then {done: {...}}. `cancelled()` is asked after every token. */
export function* stream(model, tok, prompt, params, cancelled = () => false, now = () => performance.now()) {
  const t0 = now();
  const seed = params.seed ?? Math.floor(Math.random() * 2 ** 31);
  const random = rng(seed);
  const ids = [tok.eotId, ...tok.encode(prompt)];
  const decoder = new TextDecoder("utf-8");
  let n = 0, first = null, reason = "length";
  // as in the API: a setting left out takes its default, null turns it off (so topK: null is no top-k)
  const settings = { temperature: params.temperature ?? 0.8, topK: "topK" in params ? params.topK : 50, topP: params.topP ?? null,
                     minP: params.minP ?? null, repetitionPenalty: params.repetitionPenalty ?? 1, eotId: tok.eotId };
  for (const t of iterTokens(model, ids, params.maxNewTokens ?? 120, settings, random)) {
    if (first === null) first = (now() - t0) / 1000;
    if (t === tok.eotId) { reason = "stop"; break; }
    n += 1;
    const text = decoder.decode(tok.vocab[t], { stream: true });   // holds back half a character
    if (text) yield { text };
    if (cancelled()) { reason = "cancelled"; break; }
  }
  const tail = decoder.decode();
  if (tail) yield { text: tail };
  yield { done: { tokens: n, finishReason: reason, seed, seconds: (now() - t0) / 1000, firstTokenSeconds: first } };
}

// ------------------------------------------------------------------ parse trees (top.py)
const splitWords = (s) => s.split(new RegExp(`[${PY_SPACE}]+`, "u")).filter(Boolean);

/** '[IN:X word [SL:Y word ] ]' -> {label, children}; throws for anything that is not a valid tree. */
export function parseTree(s) {
  const stack = [];
  let root = null;
  for (const tok of splitWords(s)) {
    if (tok.startsWith("[") && tok.length > 1) {
      const node = { label: tok.slice(1), children: [] }, kind = node.label.slice(0, 3);
      if ((kind !== "IN:" && kind !== "SL:") || node.label.length === 3) throw new Error(`bad label ${tok}`);
      if (stack.length) {
        if (stack[stack.length - 1].label.slice(0, 3) === kind) throw new Error(`${node.label} directly inside its kind`);
        stack[stack.length - 1].children.push(node);
      } else if (root === null) {
        if (kind !== "IN:") throw new Error("the root must be an intent");
        root = node;
      } else throw new Error("more than one root");
      stack.push(node);
    } else if (tok === "]") {
      if (!stack.length) throw new Error("unbalanced ]");
      stack.pop();
    } else {
      if (!stack.length) throw new Error(`word ${tok} outside the tree`);
      stack[stack.length - 1].children.push(tok);
    }
  }
  if (root === null || stack.length) throw new Error("empty or unclosed tree");
  return root;
}

export function treeToString(node) {
  return ["[" + node.label, ...node.children.map((c) => (typeof c === "string" ? c : treeToString(c))), "]"].join(" ");
}

export function decouple(node) {
  if (node.label.startsWith("IN:")) {
    return { label: node.label, children: node.children.filter((c) => typeof c !== "string").map(decouple) };
  }
  return { label: node.label, children: node.children.map((c) => (typeof c === "string" ? c : decouple(c))) };
}

export function scoredForm(pred) {
  try { return treeToString(decouple(parseTree(pred))); } catch { return pred; }
}

function words(node) {
  return node.children.flatMap((c) => (typeof c === "string" ? [c] : words(c)));
}

export function slotItems(node) {
  const out = [];
  for (const c of node.children) {
    if (typeof c === "string") continue;
    if (c.label.startsWith("SL:")) out.push([c.label, words(c).join(" ")]);
    out.push(...slotItems(c));
  }
  return out;
}

export function topIntent(parse) {
  const first = splitWords(parse)[0];
  return first && first.startsWith("[IN:") ? first.slice(1) : null;
}

/** Every word of an annotated tree with its innermost slot, neighbours with the same slot merged. */
export function wordSpans(tree) {
  const flat = [];
  const walk = (node, slot) => {
    const here = node.label.startsWith("SL:") ? node.label.slice(3) : slot;
    for (const c of node.children) {
      if (typeof c === "string") flat.push([c, here]); else walk(c, here);
    }
  };
  walk(tree, null);
  const spans = [];
  for (const [word, label] of flat) {
    if (spans.length && spans[spans.length - 1].label === label) spans[spans.length - 1].text += " " + word;
    else spans.push({ text: word, label });
  }
  return spans;
}

/** Labels back to upper case: "[in:create_alarm" -> "[IN:CREATE_ALARM" (top.from_model). */
export function fromModel(text, labels = "lower") {
  if (labels === "upper") return text;
  return splitWords(text).map((t) => (t.startsWith("[") ? t.toUpperCase() : t)).join(" ");
}

/** M5's parser (engine.Parser), one request at a time: greedy decoding, the request as the prompt. */
export class Parser {
  constructor(model, tok, task, { maxRequestTokens = 64, maxNewTokens = 192 } = {}) {
    if (task.name !== "hinglish-top") throw new Error("not a parser bundle");
    Object.assign(this, { model, tok, labels: task.labels, target: task.target, maxRequestTokens, maxNewTokens });
  }

  prompt(text) {
    text = normalize(text);
    if (!text) throw new Error("the request is empty");
    const ids = [this.tok.eotId, ...this.tok.encode(text), ...this.tok.encode(" =>")];
    if (ids.length > this.maxRequestTokens) {
      throw new Error(`the request is ${ids.length} tokens long; the limit is ${this.maxRequestTokens}`);
    }
    return ids;
  }

  parse(text) {
    const request = normalize(text);
    let ids;
    try { ids = this.prompt(text); } catch (e) {
      return { text: request, written: "", parse: "", valid: false, intent: null, slots: [], spans: [], tokens: 0, error: e.message };
    }
    const n = Math.max(1, Math.min(this.maxNewTokens, this.model.T - ids.length + 1));
    const out = [];
    for (const t of iterTokens(this.model, ids, n, { temperature: 0, repetitionPenalty: 1, eotId: this.tok.eotId })) {
      if (t === this.tok.eotId) break;
      out.push(t);
    }
    const written = fromModel(pyStrip(this.tok.decode(out)), this.labels);
    const annotated = this.target === "annotated";
    const parse = annotated ? scoredForm(written) : written;
    const result = { text: request, written, parse, valid: false, intent: null, slots: [], spans: [], tokens: out.length, error: null };
    let tree;
    try { tree = parseTree(parse); } catch { return result; }
    result.valid = true;
    result.intent = topIntent(parse).slice(3);
    result.slots = slotItems(tree).map(([label, w]) => ({ label: label.slice(3), text: w }));
    if (annotated) result.spans = wordSpans(parseTree(written));
    return result;
  }
}
