// The page: tabs, controls and rendering. The models run in worker.js, so the page stays responsive
// while they download and think.
import { MODEL_BASE, MODEL_PAGE } from "./config.js";

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

const PROMPTS = ["हिन्दी भाषा", "भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje"];
const REQUESTS = ["kal subah 7 baje ka alarm laga do", "mujhe Delhi ka weather batao",
                  "agle hafte mummy ko call karne ka reminder set karo", "Arijit Singh ke gaane bajao",
                  "office se ghar pahunchne mein kitna time lagega"];

const ready = { lm: false, parser: false };
const loading = { lm: false, parser: false };
let generating = null, nextId = 1;
$("model-link").href = MODEL_PAGE;

// ------------------------------------------------------------------ tabs
const tabs = [...document.querySelectorAll('[role="tab"]')];
function select(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
  if (tab.id === "tab-parse") load("parser");
}
tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => select(tab));
  tab.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    const next = tabs[(i + step + tabs.length) % tabs.length];
    select(next);
    next.focus();
  });
});

// ------------------------------------------------------------------ downloading the models
const status = { lm: $("lm-status"), parser: $("parser-status") };
const mb = (bytes) => (bytes / 1e6).toFixed(0);

function setStatus(name, text, progress = null, error = false) {
  const el = status[name];
  el.classList.toggle("error", error);
  el.textContent = text;
  if (progress) {
    const bar = document.createElement("progress");
    bar.max = progress.total;
    bar.value = progress.got;
    el.append(bar);
  }
}

function load(name) {
  if (ready[name] || loading[name]) return;
  loading[name] = true;
  setStatus(name, "Loading the model…");
  worker.postMessage({ type: "load", name });
}

// the scores shown on the page come from the bundles' own config.json
const pct = (x) => `${(100 * x).toFixed(1)}%`;
async function showScores() {
  const base = new URL(MODEL_BASE, import.meta.url);
  try {
    const [lm, parser] = await Promise.all(["lm", "parser"].map(async (n) => (await fetch(new URL(`${n}/config.json`, base))).json()));
    const fact = (key, text) => { const el = document.querySelector(`[data-fact="${key}"]`); if (el) el.textContent = text; };
    fact("params", `${(lm.params / 1e6).toFixed(1)} million`);
    fact("bpb", lm.scores.val_bpb.all.toFixed(3));
    fact("em", pct(parser.scores.test.exact_match));
    fact("intent", pct(parser.scores.test.intent_acc));
    $("parser-note").textContent = "A copy of the model fine-tuned on Hinglish-TOP finds what the request asks for (the " +
      "intent) and the details an assistant would need (the slots). On requests it never saw, the whole parse is right " +
      `${pct(parser.scores.test.exact_match)} of the time and the intent ${pct(parser.scores.test.intent_acc)}.`;
  } catch { /* the facts keep their placeholders */ }
}

// ------------------------------------------------------------------ decoding settings
const knob = (id) => $(id);
const knobs = ["temperature", "top-k", "top-p", "min-p", "repetition", "max-new"].map(knob);
function readParams() {
  const seed = $("seed").value.trim();
  return {
    temperature: Number($("temperature").value), topK: Number($("top-k").value) || null,
    topP: Number($("top-p").value) >= 1 ? null : Number($("top-p").value),
    minP: Number($("min-p").value) || null, repetitionPenalty: Number($("repetition").value),
    maxNewTokens: Number($("max-new").value), seed: seed === "" ? null : Math.min(2 ** 31 - 1, Math.max(0, Math.floor(Number(seed)))),
  };
}
function describe(p) {
  if (p.temperature === 0) return `greedy${p.repetitionPenalty !== 1 ? `, repetition penalty ${p.repetitionPenalty}` : ""}`;
  const parts = [`temperature ${p.temperature}`];
  if (p.topK) parts.push(`top-k ${p.topK}`);
  if (p.topP) parts.push(`top-p ${p.topP}`);
  if (p.minP) parts.push(`min-p ${p.minP}`);
  if (p.repetitionPenalty !== 1) parts.push(`repetition penalty ${p.repetitionPenalty}`);
  return parts.join(", ");
}
function showKnobs() {
  for (const k of knobs) document.querySelector(`output[for="${k.id}"]`).textContent = k.value;
  $("settings-summary").textContent = describe(readParams());
}
for (const k of knobs) k.addEventListener("input", showKnobs);
showKnobs();

// ------------------------------------------------------------------ examples
function examples(container, items, pick) {
  for (const text of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", () => pick(text));
    container.append(b);
  }
}
examples($("prompt-examples"), PROMPTS, (t) => { $("prompt").value = t; generate(); });
examples($("request-examples"), REQUESTS, (t) => { $("request").value = t; parse(); });

// ------------------------------------------------------------------ write
const output = $("output");
$("go").addEventListener("click", generate);
$("stop").addEventListener("click", () => worker.postMessage({ type: "stop" }));
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate(); });
$("show-tokens").addEventListener("change", (e) => output.classList.toggle("show-tokens", e.target.checked));

function generate() {
  if (!ready.lm || generating) return;
  generating = nextId++;
  const prompt = document.createElement("span");
  prompt.className = "prompt";
  prompt.textContent = $("prompt").value;
  output.replaceChildren(prompt);
  $("stats").textContent = "";
  $("go").disabled = true;
  $("stop").disabled = false;
  worker.postMessage({ type: "generate", id: generating, prompt: $("prompt").value, params: readParams() });
}

function statsLine(s) {
  const per = (s.seconds - (s.firstTokenSeconds ?? 0)) / Math.max(s.tokens - 1, 1) * 1000;
  const end = { stop: " The model ended the text.", cancelled: " Stopped.", length: "" }[s.finishReason];
  return `${s.tokens} tokens in ${s.seconds.toFixed(1)} s, ${per.toFixed(0)} ms per token.${end} Seed ${s.seed}.`;
}

// ------------------------------------------------------------------ understand a request
$("parse").addEventListener("click", parse);
$("request").addEventListener("keydown", (e) => { if (e.key === "Enter") parse(); });

function parse() {
  if (!ready.parser || $("parse").dataset.busy) return;
  $("parse").dataset.busy = "1";
  $("parse").disabled = true;
  worker.postMessage({ type: "parse", id: nextId++, text: $("request").value });
}

const human = (label) => label.toLowerCase().replaceAll("_", " ");
function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function renderParse(r, seconds) {
  const box = $("parsed");
  box.replaceChildren();
  if (r.error) { box.append(element("p", "intent", `This request can't be parsed: ${r.error}.`)); return; }
  if (!r.valid) {
    box.append(element("p", "intent", "The model didn't write a well-formed parse this time. What it wrote:"),
               element("pre", "tree", r.written));
    return;
  }
  const sentence = element("div", "annotated"), colors = new Map();
  for (const span of r.spans) {
    if (!span.label) { for (const w of span.text.split(" ")) sentence.append(element("span", "word", w)); continue; }
    if (!colors.has(span.label)) colors.set(span.label, `var(--slot-${(colors.size % 5) + 1})`);
    const slot = element("span", "slot");
    slot.style.background = colors.get(span.label);
    slot.append(element("span", "", span.text), element("small", "", human(span.label)));
    sentence.append(slot);
  }
  const intent = element("p", "intent", "Intent: ");
  intent.append(element("strong", "", human(r.intent)));
  box.append(sentence, intent, element("pre", "tree", r.written),
             element("p", "parse-meta", `Parsed on your device in ${seconds.toFixed(1)} s; the model wrote ${r.tokens} tokens.`));
}

// ------------------------------------------------------------------ messages from the worker
worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "progress":
      setStatus(data.name, data.total ? `Downloading the model into your browser: ${mb(data.got)} of ${mb(data.total)} MB`
                                      : `Downloading the model into your browser: ${mb(data.got)} MB`, data.total ? data : null);
      break;
    case "ready":
      ready[data.name] = true;
      loading[data.name] = false;
      setStatus(data.name, "");
      if (data.name === "lm") $("go").disabled = false;
      else { $("parse").disabled = false; }
      break;
    case "piece":
      if (data.id === generating) output.append(element("span", "tok", data.text));
      break;
    case "done":
      if (data.id !== generating) break;
      generating = null;
      $("go").disabled = false;
      $("stop").disabled = true;
      $("stats").textContent = statsLine(data.stats);
      break;
    case "parsed":
      delete $("parse").dataset.busy;
      $("parse").disabled = false;
      renderParse(data.result, data.seconds);
      break;
    case "error":
      loading[data.name] = false;
      setStatus(data.name, ready[data.name] ? `Something went wrong: ${data.message}. Reload the page to start again.`
                                            : `Couldn't load the model: ${data.message}. Reload the page to try again.`, null, true);
      if (data.name === "lm") { generating = null; $("go").disabled = !ready.lm; $("stop").disabled = true; }
      else { delete $("parse").dataset.busy; $("parse").disabled = !ready.parser; }
      break;
  }
};

load("lm");          // start at once: the first click shouldn't wait for the download
showScores();
