"""Load test for the service (M6): how latency and throughput change as more clients arrive at once.

    python scripts/load_test.py                      # starts its own servers, ~6 minutes
    python scripts/load_test.py --url http://localhost:8000 --only generate   # a running server

Every client sends a request, waits for the answer, and sends the next ("closed loop"), so
`concurrency` = requests in flight. Measured on the client, as a user would see it:

  parse     requests per second and latency, at 1-32 clients, with batching off (TINYGPT_MAX_BATCH=1)
            and on (16) — the server is started once per setting
  generate  time to first token and the whole answer for 64-token streams, at 1-8 clients
  overload  40 streams at once against a queue of 8: how many are refused, and how fast

Writes docs/results/m6_load.json (every measurement), m6_load.md (tables) and m6_parse.png;
--replot redraws the tables and the chart from the JSON without measuring again.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tinygpt import top  # noqa: E402

PROMPTS = ["भारत की राजधानी", "महात्मा गांधी का जन्म", "yaar kal movie dekhne", "mujhe kal subah 7 baje"]


# ------------------------------------------------------------------ a server to test
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """`python -m tinygpt.serve --no-ui` in a subprocess, with TINYGPT_* settings."""

    def __init__(self, settings: dict, log_path: Path):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        env = {**os.environ, **{f"TINYGPT_{k.upper()}": str(v) for k, v in settings.items()}}
        self.log = open(log_path, "w")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen([sys.executable, "-m", "tinygpt.serve", "--no-ui", "--port", str(self.port)],
                                     env=env, stdout=self.log, stderr=subprocess.STDOUT, cwd=ROOT)
        while True:
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None or time.perf_counter() - t0 > 180:
                raise SystemExit(f"server did not start; see {log_path}")
            time.sleep(0.2)
        self.ready_seconds = time.perf_counter() - t0

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


# ------------------------------------------------------------------ clients
def summary(latencies: list[float]) -> dict:
    a = np.asarray(latencies) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())} if len(a) else {}


async def closed_loop(concurrency: int, seconds: float, one_request, warmup: float = 2.0) -> list:
    """`concurrency` clients sending requests back to back. Returns the requests that *finished*
    inside a `seconds`-long window, which opens once every client has had one answer (and at least
    `warmup` s after the start). Before that, the clients' first requests all arrived at the same
    moment — a burst, not the steady state being measured. Counting finishes in a fixed window gives
    the true rate even when one request takes longer than the window."""
    results: list = []
    start = time.perf_counter()
    window: dict[str, float] = {}
    answered = [0]

    async def client(i: int) -> None:
        n = 0
        while "stop" not in window or time.perf_counter() < window["stop"]:
            r = await one_request(i, n)
            now = time.perf_counter()
            if n == 0:
                answered[0] += 1
                if answered[0] == concurrency:             # everyone has had a turn: open the window
                    window["start"] = max(now, start + warmup)
                    window["stop"] = window["start"] + seconds
            if "start" in window and window["start"] <= now <= window["stop"]:
                results.append(r)
            n += 1

    await asyncio.gather(*(client(i) for i in range(concurrency)))
    return results


async def parse_level(url: str, texts: list[str], concurrency: int, seconds: float) -> dict:
    async with httpx.AsyncClient(base_url=url, timeout=60, limits=httpx.Limits(max_connections=64)) as c:
        async def one(i, n):
            t0 = time.perf_counter()
            r = await c.post("/v1/parse", json={"text": texts[(i * 997 + n) % len(texts)]})
            r.raise_for_status()
            return time.perf_counter() - t0, r.json()["batch_size"]

        results = await closed_loop(concurrency, seconds, one)
    lat = [r[0] for r in results]
    return {"concurrency": concurrency, "requests": len(results), "per_second": len(results) / seconds,
            "mean_batch": float(np.mean([r[1] for r in results])), **summary(lat)}


async def stream_once(c: httpx.AsyncClient, body: dict) -> dict:
    """One streamed generation: client-side time to first token, total time, tokens."""
    t0 = time.perf_counter()
    first, tokens = None, 0
    async with c.stream("POST", "/v1/generate/stream", json=body) as r:
        if r.status_code != 200:
            await r.aread()
            return {"status": r.status_code, "seconds": time.perf_counter() - t0}
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "token":
                first = first or time.perf_counter() - t0
            elif line.startswith("data: ") and event == "done":
                tokens = json.loads(line[6:])["tokens"]
    return {"status": 200, "first": first, "seconds": time.perf_counter() - t0, "tokens": tokens}


async def generate_level(url: str, concurrency: int, seconds: float, new_tokens: int) -> dict:
    async with httpx.AsyncClient(base_url=url, timeout=120, limits=httpx.Limits(max_connections=64)) as c:
        async def one(i, n):
            return await stream_once(c, {"prompt": PROMPTS[(i + n) % len(PROMPTS)], "max_new_tokens": new_tokens,
                                         "seed": 1000 * i + n})

        results = await closed_loop(concurrency, seconds, one)
    ok = [r for r in results if r["status"] == 200]
    ttft = summary([r["first"] for r in ok if r["first"] is not None])
    total = summary([r["seconds"] for r in ok])
    return {"concurrency": concurrency, "requests": len(ok), "rejected": len(results) - len(ok),
            "tokens_per_second": sum(r["tokens"] for r in ok) / seconds,
            "ttft_p50_ms": ttft.get("p50_ms"), "ttft_p95_ms": ttft.get("p95_ms"),
            "total_p50_ms": total.get("p50_ms"), "total_p95_ms": total.get("p95_ms")}


async def overload(url: str, n: int, new_tokens: int) -> dict:
    async with httpx.AsyncClient(base_url=url, timeout=300, limits=httpx.Limits(max_connections=n + 4)) as c:
        results = await asyncio.gather(*(stream_once(c, {"prompt": PROMPTS[i % 4], "max_new_tokens": new_tokens,
                                                         "seed": i}) for i in range(n)))
    served = [r for r in results if r["status"] == 200]
    refused = [r for r in results if r["status"] == 503]
    return {"sent": n, "served": len(served), "refused": len(refused),
            "refused_p50_ms": summary([r["seconds"] for r in refused]).get("p50_ms"),
            "served_last_ms": max(r["seconds"] for r in served) * 1000 if served else None}


# ------------------------------------------------------------------ report
def plot_parse(res: dict, path: Path) -> None:
    """Requests per second and p95 latency against clients, batching off and on: two panels, one
    measure each (never two y-scales on one axis), the same colour for a setting in both."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tinygpt.plotting import INK, INK2, SURFACE, run_colors, style_axes

    names = {"off": "batching off", "on": "batching on"}
    colors = run_colors(list(names.values()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    panels = [("per_second", "requests per second (higher is better)"),
              ("p95_ms", "95th-percentile latency, ms (lower is better)")]
    for ax, (key, title) in zip(axes, panels):
        style_axes(ax)
        for setting, rows in res["parse"].items():
            name = names.get(setting, setting)
            xs, ys = [r["concurrency"] for r in rows], [r[key] for r in rows]
            ax.plot(xs, ys, color=colors.get(name, INK), linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([r["concurrency"] for r in next(iter(res["parse"].values()))])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("clients sending requests at the same time", color=INK2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(right=max(r["concurrency"] for rows in res["parse"].values() for r in rows) * 2.2)  # end labels
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=9)
    fig.suptitle("Parsing: one request per forward pass vs everything waiting in one batch",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(res: dict, out: Path) -> None:
    lines = ["# M6 — load test", "",
             f"{res['machine']}; server: {res.get('server', {}).get('threads', '?')} torch threads, "
             f"device {res.get('server', {}).get('device', '?')}. Client and server on the same machine; closed "
             f"loop (each client waits for its answer before sending the next). Each row counts the requests that "
             f"finish in a {res['seconds']:.0f}-second window, opened once every client has had its first answer. "
             "Produced by `scripts/load_test.py`.", ""]
    if "parse" in res:
        lines += ["## Parsing: batching off vs on", "",
                  "Each request is one Hinglish-TOP validation request. *Batching on*: the worker parses everything "
                  "waiting in its queue together (up to 16) in one padded forward pass.", "",
                  "| clients | batching | requests/s | mean batch | p50 ms | p95 ms | p99 ms |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for setting, rows in res["parse"].items():
            for r in rows:
                lines.append(f"| {r['concurrency']} | {setting} | {r['per_second']:.1f} | {r['mean_batch']:.1f} | "
                             f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} |")
        lines.append("")
    if "generate" in res:
        lines += ["## Generation: one stream at a time", "",
                  f"{res['new_tokens']}-token streams (temperature 0.8, top-k 50). The worker serves one stream at a "
                  "time; the others wait in its queue.", "",
                  "| clients | streams | tokens/s (all clients) | first token p50 ms | p95 ms | whole stream p50 ms | p95 ms |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in res["generate"]:
            lines.append(f"| {r['concurrency']} | {r['requests']} | {r['tokens_per_second']:.0f} | "
                         f"{r['ttft_p50_ms']:.0f} | {r['ttft_p95_ms']:.0f} | {r['total_p50_ms']:.0f} | "
                         f"{r['total_p95_ms']:.0f} |")
        lines.append("")
    if "overload" in res:
        o = res["overload"]
        lines += ["## Overload", "",
                  f"{o['sent']} streams sent at the same moment to a server whose generation queue holds 8: "
                  f"{o['served']} served (the last finished after {o['served_last_ms'] / 1000:.1f} s), "
                  f"{o['refused']} refused with HTTP 503 in {o['refused_p50_ms']:.0f} ms (median).", ""]
    if "ready_seconds" in res:
        lines += [f"Server start to first healthy response (Python imports + loading both models): "
                  f"{res['ready_seconds']:.1f} s.", ""]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="test this running server instead of starting servers")
    ap.add_argument("--only", choices=["parse", "generate", "overload"])
    ap.add_argument("--seconds", type=float, default=10.0, help="length of each measurement window")
    ap.add_argument("--parse-levels", default="1,2,4,8,16,32")
    ap.add_argument("--generate-levels", default="1,2,4,8")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--data-dir", default="data/top")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--replot", action="store_true", help="redraw from m6_load.json, measure nothing")
    args = ap.parse_args()

    out = Path(args.out)
    if args.replot:
        res = json.loads((out / "m6_load.json").read_text())
        write_report(res, out / "m6_load.md")
        if "parse" in res:
            plot_parse(res, out / "m6_parse.png")
        print(f"-> {out / 'm6_load.md'}, {out / 'm6_parse.png'}")
        return

    texts = [e.query for e in top.load("val", args.data_dir)]
    random.Random(0).shuffle(texts)
    out.mkdir(parents=True, exist_ok=True)
    res = {"machine": f"{platform.machine()} {platform.system()}, {os.cpu_count()} CPU cores",
           "seconds": args.seconds, "new_tokens": args.new_tokens}
    p_levels = [int(x) for x in args.parse_levels.split(",")]
    g_levels = [int(x) for x in args.generate_levels.split(",")]
    run = lambda part: args.only in (None, part)   # noqa: E731

    def measure(url: str, parts: set, label: str = "on") -> None:
        res["server"] = {k: httpx.get(url + "/v1/info").json()[k] for k in ("threads", "device", "torch")}
        if "parse" in parts:
            res.setdefault("parse", {})[label] = []
            for c in p_levels:
                row = asyncio.run(parse_level(url, texts, c, args.seconds))
                res["parse"][label].append(row)
                print(f"parse, batching {label}, {c:>2} clients: {row['per_second']:6.1f} req/s, "
                      f"batch {row['mean_batch']:4.1f}, p50 {row['p50_ms']:5.0f} ms, p95 {row['p95_ms']:5.0f} ms")
        if "generate" in parts:
            res["generate"] = []
            for c in g_levels:
                row = asyncio.run(generate_level(url, c, args.seconds * 1.5, args.new_tokens))
                res["generate"].append(row)
                print(f"generate, {c} clients: {row['tokens_per_second']:5.0f} tokens/s, first token p50 "
                      f"{row['ttft_p50_ms']:5.0f} ms, p95 {row['ttft_p95_ms']:5.0f} ms")
        if "overload" in parts:
            res["overload"] = asyncio.run(overload(url, 40, args.new_tokens))
            print(f"overload: {res['overload']}")

    if args.url:
        measure(args.url, {p for p in ("parse", "generate", "overload") if run(p)}, "as configured")
    else:
        logs = Path("runs/m6")
        logs.mkdir(parents=True, exist_ok=True)
        if run("parse"):
            server = Server({"max_batch": 1}, logs / "server-batch1.log")
            measure(server.url, {"parse"}, "off")
            server.stop()
        server = Server({"max_batch": 16}, logs / "server.log")
        res["ready_seconds"] = server.ready_seconds
        measure(server.url, {p for p in ("parse", "generate", "overload") if run(p)}, "on")
        server.stop()

    (out / "m6_load.json").write_text(json.dumps(res, indent=2))
    write_report(res, out / "m6_load.md")
    if "parse" in res:
        plot_parse(res, out / "m6_parse.png")
    print(f"-> {out / 'm6_load.json'}, {out / 'm6_load.md'}, {out / 'm6_parse.png'}")


if __name__ == "__main__":
    main()
