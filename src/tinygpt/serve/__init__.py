"""M6: serving TinyGPT.

    engine.py   the two models as plain Python objects (no web code)       — used everywhere
    worker.py   one thread per model, a bounded queue, batching, cancelling — the HTTP server
    api.py      the FastAPI app: JSON + streaming endpoints, metrics        — needs `.[serve]`
    ui.py       the Gradio demo                                              — server and Space

Importing this package loads nothing heavy; each module imports only what it needs.
"""
