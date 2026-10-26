// The page's second thread: downloads the models once, then runs them (web/tinygpt.js) without
// freezing the page. Messages in: load, generate, stop, parse. Messages out: progress, ready,
// piece, done, parsed, error.
import { MODEL_BASE } from "./config.js";
import { GPT, Parser, Tokenizer, readSafetensors, stream } from "./tinygpt.js";

const CACHE = "tinygpt-models-v1";
const loaded = {};                          // "lm" | "parser" -> Promise of {config, tok, model}
let stopRequested = false;
const post = (message) => self.postMessage(message);

async function openCache() {               // Cache Storage: kept between visits (absent in some private windows)
  try { return await caches.open(CACHE); } catch { return null; }
}

async function fetchOk(url) {
  let res;
  try { res = await fetch(url); } catch { throw new Error("the network request failed"); }
  if (!res.ok) throw new Error(`the server answered HTTP ${res.status}`);
  return res;
}

/** The file's bytes, reporting progress as they arrive. */
async function download(url, onProgress) {
  const res = await fetchOk(url);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  return bytes.buffer;
}

async function sha256(buffer) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", buffer));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** A bundle as bundle.load_bundle reads it: config, tokenizer, then weights checked against their
 *  SHA-256. The weights are cached only once they pass, and checked again when read from the cache. */
async function loadBundle(name) {
  const base = new URL(`${name}/`, new URL(MODEL_BASE, self.location.href)).href;
  const config = await (await fetchOk(base + "config.json")).json();
  const tok = new Tokenizer(await (await fetchOk(base + config.tokenizer)).json());
  const url = base + "model.safetensors", want = config.sha256["model.safetensors"];
  const cache = await openCache();
  let buffer = null;
  const hit = cache && await cache.match(url);
  if (hit) {
    buffer = await hit.arrayBuffer();
    if (await sha256(buffer) !== want) { buffer = null; await cache.delete(url); }   // damaged: download again
  }
  if (!buffer) {
    buffer = await download(url, (got, total) => post({ type: "progress", name, got, total }));
    if (await sha256(buffer) !== want) throw new Error("the downloaded weights don't match their checksum");
    cache?.put(url, new Response(buffer)).catch(() => {});        // the Response keeps its own copy
  }
  const model = new GPT(config.model_config, readSafetensors(buffer));
  post({ type: "ready", name });
  return { config, tok, model };
}

function load(name) {        // one download per model; after a failure, the next request tries again
  loaded[name] ??= loadBundle(name).catch((e) => { delete loaded[name]; throw e; });
  return loaded[name];
}

// a zero-delay pause, so a "stop" message can arrive between two tokens
const channel = new MessageChannel();
const tick = () => new Promise((resolve) => { channel.port1.onmessage = () => resolve(); channel.port2.postMessage(0); });

async function generate({ id, prompt, params }) {
  const { model, tok } = await load("lm");
  stopRequested = false;
  for (const event of stream(model, tok, prompt, params, () => stopRequested)) {
    post(event.done ? { type: "done", id, stats: event.done } : { type: "piece", id, text: event.text });
    await tick();
  }
}

async function parse({ id, text }) {
  const { model, tok, config } = await load("parser");
  const t0 = performance.now();
  const result = new Parser(model, tok, config.task).parse(text);
  post({ type: "parsed", id, result, seconds: (performance.now() - t0) / 1000 });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.name);
    else if (data.type === "stop") stopRequested = true;
    else if (data.type === "generate") await generate(data);
    else if (data.type === "parse") await parse(data);
  } catch (e) {
    post({ type: "error", id: data.id, name: data.name ?? (data.type === "parse" ? "parser" : "lm"), message: e.message });
  }
};
