"""Run the service: the API and the Gradio demo on one port.

    python -m tinygpt.serve                     # http://127.0.0.1:8000  (demo at /, API docs at /docs)
    python -m tinygpt.serve --host 0.0.0.0      # reachable from other machines (Docker does this)
    TINYGPT_MAX_BATCH=1 python -m tinygpt.serve # any setting: TINYGPT_<NAME>, see api.Settings

One process, one copy of each model. More traffic = more processes or machines behind a load
balancer, each with its own models — not uvicorn --workers inside one container, which would
load the models once per worker and split the queue so that no worker sees the whole load.
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--no-ui", action="store_true", help="API only (same as TINYGPT_UI=0)")
    args = ap.parse_args()
    if args.no_ui:
        os.environ["TINYGPT_UI"] = "0"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # our middleware logs every API request with its time and id, so uvicorn's access log is off
    uvicorn.run("tinygpt.serve.api:create_app", factory=True, host=args.host, port=args.port,
                access_log=False, timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
