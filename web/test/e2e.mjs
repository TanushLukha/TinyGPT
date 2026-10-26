// The demo page in a real browser (Chromium, driven by Playwright) — what tests/test_web.py can't
// see: the page downloads both models into a worker, writes, stops, parses and reports errors, and
// the deployed page (weights from huggingface.co, redirected to a file CDN) works only with CORS.
//
//     npm install --no-save playwright && npx playwright install chromium     # once
//     node web/test/e2e.mjs                      # needs models/ (scripts/export.py); ~1 minute
//     node web/test/e2e.mjs --shots shots/       # also save screenshots
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { createReadStream, mkdirSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const shotsAt = process.argv.indexOf("--shots");
const SHOTS = shotsAt > 0 ? process.argv[shotsAt + 1] : null;
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
                ".json": "application/json", ".md": "text/markdown" };

/** A static file server on a free port. `cors`: send Access-Control-Allow-Origin; `slow`: send
 *  .safetensors files at ~4 MB/s; `prefix`: a path prefix to strip (the CDN serves /xet/...). */
function serve(root, { cors = false, slow = false, prefix = "" } = {}) {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname).replace(prefix, "/"));
    let file = join(root, path);
    try { if (statSync(file).isDirectory()) file = join(file, "index.html"); statSync(file); } catch {
      res.writeHead(404).end("not found"); return;
    }
    const headers = { "content-type": TYPES[extname(file)] ?? "application/octet-stream",
                      "content-length": statSync(file).size };
    if (cors) headers["access-control-allow-origin"] = "*";
    res.writeHead(200, headers);
    const stream = createReadStream(file, { highWaterMark: 64 * 1024 });
    if (slow && file.endsWith(".safetensors")) {
      stream.on("data", (chunk) => { stream.pause(); res.write(chunk); setTimeout(() => stream.resume(), 16); });
      stream.on("end", () => res.end());
    } else stream.pipe(res);
    res.on("close", () => stream.destroy());
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` })));
}

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? `: ${detail}` : ""}`);
  if (!ok) failures += 1;
}
const enabled = (page, id) => page.waitForFunction((i) => !document.getElementById(i).disabled, id, { timeout: 120000 });
const failed = (page) => page.waitForFunction(() => document.querySelector("#lm-status.error"), null, { timeout: 60000 });
const statusText = (page) => page.$eval("#lm-status", (e) => e.textContent);
const shot = (page, name) => SHOTS && page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: true });

async function written(page) {
  await page.waitForFunction(() => document.getElementById("stats").textContent !== "", null, { timeout: 120000 });
  return { text: await page.$eval("#output", (e) => e.textContent), stats: await page.$eval("#stats", (e) => e.textContent) };
}
async function generate(page) {
  await page.$eval("#stats", (e) => { e.textContent = ""; });
  await page.click("#go");
  return written(page);
}
async function parsed(page) {
  await page.waitForFunction(() => document.querySelector("#parsed .parse-meta, #parsed .intent"), null, { timeout: 60000 });
  return page.$eval("#parsed", (e) => ({ text: e.innerText, tree: e.querySelector(".tree")?.textContent ?? "" }));
}
const setRange = (page, id, value) => page.$eval(`#${id}`, (e, v) => { e.value = v; e.dispatchEvent(new Event("input")); }, value);

const site = await serve(ROOT);
const slowSite = await serve(ROOT, { slow: true });
const cdn = await serve(join(ROOT, "models"), { cors: true, prefix: "/xet/" });
const cdnNoCors = await serve(join(ROOT, "models"), { prefix: "/xet/" });
const browser = await chromium.launch();
const page_ = `${site.url}/web/`;

// 1. the page, as python -m http.server would serve it
{
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let t0 = Date.now();
  await page.goto(page_);
  await enabled(page, "go");
  check("language model downloaded and ready", true, `${((Date.now() - t0) / 1000).toFixed(1)} s`);

  await page.click(".settings summary");
  await page.fill("#seed", "42");
  const a = await generate(page), b = await generate(page);
  check("writes text", a.text.length > "हिन्दी भाषा".length, a.stats);
  check("the same seed writes the same text", a.text === b.text);
  check("the settings summary follows the sliders", (await page.$eval("#settings-summary", (e) => e.textContent)) === "temperature 0.8, top-k 50");
  await shot(page, "1-write");

  await page.fill("#seed", "");
  await setRange(page, "max-new", 256);
  await page.$eval("#stats", (e) => { e.textContent = ""; });
  await page.click("#go");
  await page.waitForTimeout(300);
  await page.click("#stop");
  const stopped = await written(page);
  check("Stop ends the text early", stopped.stats.includes("Stopped"), stopped.stats);

  await setRange(page, "temperature", 0);
  check("temperature 0 is greedy", (await page.$eval("#settings-summary", (e) => e.textContent)) === "greedy");

  t0 = Date.now();
  await page.click("#tab-parse");
  await enabled(page, "parse");
  check("parser downloaded when its tab opens", true, `${((Date.now() - t0) / 1000).toFixed(1)} s`);
  await page.click("#parse");
  const p = await parsed(page);
  check("parses a request", p.tree.startsWith("[IN:") && p.text.includes("Intent:"), p.tree);
  await shot(page, "2-parse");
  await page.$eval("#parsed", (e) => e.replaceChildren());
  await page.fill("#request", "   ");
  await page.click("#parse");
  check("an empty request gets a message", (await parsed(page)).text.includes("the request is empty"));

  await page.click("#tab-about");
  const facts = await page.$$eval("#facts td", (tds) => tds.map((td) => td.textContent));
  check("About shows the bundles' scores", facts.every((f) => f !== "–"), facts.join(" | "));
  await page.focus("#tab-about");
  await page.keyboard.press("ArrowRight");
  check("arrow keys move between tabs", (await page.evaluate(() => document.activeElement.id)) === "tab-write");

  t0 = Date.now();
  await page.reload();
  await enabled(page, "go");
  check("second visit loads from the cache", Date.now() - t0 < 5000, `${((Date.now() - t0) / 1000).toFixed(1)} s`);
  check("no errors in the page", errors.length === 0, errors.join("; "));

  await page.emulateMedia({ colorScheme: "dark" });
  await shot(page, "3-dark");
  await context.close();
}

// 2. a phone
{
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true });
  const page = await context.newPage();
  await page.goto(page_);
  await enabled(page, "go");
  await generate(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check("fits a 390-pixel-wide phone", overflow <= 0, overflow > 0 ? `${overflow}px too wide` : "");
  await shot(page, "4-phone");
  await context.close();
}

// 3. progress and failures
async function fresh(url, setup) {
  const context = await browser.newContext();
  if (setup) await setup(context);
  const page = await context.newPage();
  await page.goto(url);
  return { context, page };
}
{
  const { context, page } = await fresh(`${slowSite.url}/web/`);
  await page.waitForFunction(() => document.querySelector("#lm-status progress")?.value > 4e6, null, { timeout: 60000 });
  check("shows download progress", /of 55 MB/.test(await statusText(page)), await statusText(page));
  await context.close();
}
for (const [name, fulfil, expect] of [
  ["a missing file", { status: 404, body: "not found" }, "HTTP 404"],
  ["a damaged download", { status: 200, body: Buffer.alloc(1000) }, "checksum"],
]) {
  const { context, page } = await fresh(page_, (c) => c.route("**/models/lm/model.safetensors", (r) => r.fulfill(fulfil)));
  await failed(page);
  check(`explains ${name}`, (await statusText(page)).includes(expect), await statusText(page));
  await context.close();
}
{
  const { context, page } = await fresh(page_, (c) => c.route("**/models/lm/config.json", (r) => r.abort("internetdisconnected")));
  await failed(page);
  check("explains a network failure", (await statusText(page)).includes("network request failed"), await statusText(page));
  await context.close();
}

// 4. the page as deployed: config.js names huggingface.co at a commit, which redirects the weights
//    to a CDN on another origin (here: local servers with and without CORS)
const REVISION = "0123456789abcdef0123456789abcdef01234567";
async function deployed(cdnUrl) {
  return fresh(page_, async (c) => {
    await c.route("**/web/config.js", (r) => r.fulfill({ contentType: "text/javascript", body:
      `export const MODEL_BASE = "https://huggingface.co/someone/tinygpt/resolve/${REVISION}/";\n` +
      `export const MODEL_PAGE = "https://huggingface.co/someone/tinygpt";\n` }));
    await c.route("https://huggingface.co/**", (r) => {
      const m = new URL(r.request().url()).pathname.match(/^\/someone\/tinygpt\/resolve\/([0-9a-f]{40})\/(.+)$/);
      if (!m || m[1] !== REVISION) return r.fulfill({ status: 404 });
      const cors = { "access-control-allow-origin": "*" };
      if (m[2].endsWith(".safetensors")) return r.fulfill({ status: 302, headers: { location: `${cdnUrl}/xet/${m[2]}`, ...cors } });
      return r.fulfill({ status: 200, headers: { "content-type": "application/json", ...cors }, path: join(ROOT, "models", m[2]) });
    });
  });
}
{
  const { context, page } = await deployed(cdn.url);
  await enabled(page, "go");
  const out = await generate(page);
  check("deployed: weights from the Hub, through a CDN with CORS", out.text.length > 0, out.stats);
  check("deployed: links to the model repo", (await page.$eval("#model-link", (e) => e.href)) === "https://huggingface.co/someone/tinygpt");
  await context.close();
}
{
  const { context, page } = await deployed(cdnNoCors.url);
  await failed(page);
  check("deployed: without CORS the page says the download failed", (await statusText(page)).includes("network request failed"));
  await context.close();
}

await browser.close();
for (const s of [site, slowSite, cdn, cdnNoCors]) s.server.close();
console.log(failures ? `${failures} check(s) failed` : "all checks passed");
process.exit(failures ? 1 : 0);
