"""Worker threads between the web server and the models (M6).

The server is asynchronous: one event loop handles every connection. A forward pass run inside
that loop would freeze every other request while it runs, so each model gets its own thread and
requests reach it through a bounded queue:

    request ─ submit() ─▶ queue ─▶ worker thread ─▶ model
       ▲          │ (full: Busy -> HTTP 503)    │
       └──── channel ◀── pieces / results ──────┘

* Backpressure: at most `max_queue` jobs wait. Beyond that a request fails at once with Busy
  (HTTP 503 + Retry-After) instead of waiting minutes in an ever-growing queue.
* Cancelling: a job carries an Event; the generator checks it after every token, so a client that
  disconnects stops its generation, and a job cancelled while still waiting never starts.
* Batching (parser only): the worker takes the job at the head of the queue plus whatever else is
  waiting, up to `max_batch`, and parses them in one forward pass. Under no load a request runs
  alone and immediately; under load requests pile up while the model is busy, so batches grow by
  themselves. `max_wait_ms` > 0 also waits a little for company (Triton's "max queue delay").
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .engine import Done, Parser, TextGenerator

log = logging.getLogger("tinygpt.serve")


class Busy(Exception):
    """Too many requests are already waiting: retry later (HTTP 503)."""


@dataclass
class Error:
    message: str


# ------------------------------------------------------------------ channels: worker -> requester
class AsyncChannel:
    """For a coroutine in the event loop. The worker thread may not touch asyncio objects directly,
    so every put is handed to the loop with call_soon_threadsafe."""

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:                           # the loop has closed: nobody is listening
            pass

    async def get(self):
        return await self._queue.get()


class SyncChannel:
    """For a thread (e.g. a Gradio handler) that can simply block."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()

    def put(self, item) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None):
        return self._queue.get(timeout=timeout)


@dataclass
class Job:
    payload: Any                       # (prompt, GenerateParams) or a request text
    channel: Any                       # AsyncChannel | SyncChannel
    cancel: threading.Event = field(default_factory=threading.Event)
    submitted: float = field(default_factory=time.perf_counter)
    started: float | None = None

    @property
    def queue_seconds(self) -> float:
        return (self.started or time.perf_counter()) - self.submitted


# ------------------------------------------------------------------ workers
class Worker:
    """One thread that owns a model and runs its jobs in order."""

    def __init__(self, name: str, max_queue: int, on_event: Callable[[str, dict], None] | None = None):
        self.name, self.max_queue = name, max_queue
        self.on_event = on_event or (lambda kind, data: None)     # metrics hook
        self._queue: queue.Queue = queue.Queue()                  # unbounded: the limit is enforced
        self._waiting = 0                                         # in submit(), so stop() never blocks
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name=f"tinygpt-{name}", daemon=True)

    @property
    def waiting(self) -> int:
        return self._waiting

    def start(self) -> "Worker":
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)

    def submit(self, job: Job) -> Job:
        with self._lock:
            if self._waiting >= self.max_queue:
                self.on_event("rejected", {"worker": self.name})
                raise Busy(f"{self.name}: {self._waiting} requests already waiting")
            self._waiting += 1
        self._queue.put(job)
        return job

    def _take(self, block: bool = True, timeout: float | None = None) -> Job | None:
        job = self._queue.get(block, timeout)
        if job is not None:
            with self._lock:
                self._waiting -= 1
        return job

    def _loop(self) -> None:
        with torch.inference_mode():                   # no autograd bookkeeping in this thread
            while True:
                job = self._take()
                if job is None:
                    return
                jobs = [j for j in self._collect(job) if not j.cancel.is_set()]
                if not jobs:
                    continue
                now = time.perf_counter()
                for j in jobs:
                    j.started = now
                    self.on_event("queue", {"worker": self.name, "seconds": j.queue_seconds})
                try:
                    self.handle(jobs)
                except Exception as e:                 # never kill the thread: report and go on
                    log.exception("%s failed", self.name)
                    for j in jobs:
                        j.channel.put(Error(f"{type(e).__name__}: {e}"))

    def _collect(self, first: Job) -> list[Job]:
        return [first]

    def handle(self, jobs: list[Job]) -> None:
        raise NotImplementedError


class GenerateWorker(Worker):
    """Text generation, one request at a time. (Batching generation needs *continuous* batching —
    requests joining and leaving a running batch at every token — see docs/M6_serving.md.)"""

    def __init__(self, generator: TextGenerator, max_queue: int = 8, **kw):
        super().__init__("generate", max_queue, **kw)
        self.generator = generator

    def handle(self, jobs: list[Job]) -> None:
        job = jobs[0]
        prompt, params = job.payload
        for event in self.generator.stream(prompt, params, cancelled=job.cancel.is_set):
            job.channel.put(event)
            if isinstance(event, Done):
                self.on_event("generated", {"tokens": event.tokens, "reason": event.finish_reason,
                                            "first_token_seconds": event.first_token_seconds})


class ParseWorker(Worker):
    def __init__(self, parser: Parser, max_queue: int = 64, max_batch: int = 16, max_wait_ms: float = 0.0, **kw):
        super().__init__("parse", max_queue, **kw)
        self.parser, self.max_batch, self.max_wait = parser, max_batch, max_wait_ms / 1000

    def _collect(self, first: Job) -> list[Job]:
        jobs = [first]
        deadline = time.perf_counter() + self.max_wait
        while len(jobs) < self.max_batch:
            try:
                remaining = deadline - time.perf_counter()
                job = self._take(block=remaining > 0, timeout=remaining if remaining > 0 else None)
            except queue.Empty:
                break
            if job is None:                            # stop() was called: finish this batch first
                self._queue.put(None)
                break
            jobs.append(job)
        return jobs

    def handle(self, jobs: list[Job]) -> None:
        results = self.parser.parse([j.payload for j in jobs])   # a bad request fails only itself
        self.on_event("batch", {"size": len(jobs)})
        for job, result in zip(jobs, results):
            job.channel.put(result)
