"""Publish TinyGPT on the Hugging Face Hub: the models as a model repo, the demo as a Space.

    hf auth login                   # once: paste a token with write access (huggingface.co/settings/tokens)
    python scripts/export.py        # models/ (if not done yet)
    python scripts/deploy_hf.py     # -> huggingface.co/<you>/tinygpt  and  huggingface.co/spaces/<you>/tinygpt

1. The model repo gets models/: lm/, parser/ and the model card (README.md).
2. The Space, by default a static one (--kind static, free on every account): the page in web/, with a
   config.js pointing at the model repo *at the commit just uploaded*. Visitors' browsers download the
   weights from there and run them (web/tinygpt.js), so uploading new weights later changes nothing
   until the Space is deployed again.
   --kind zerogpu instead deploys the Gradio app (space/app.py) on ZeroGPU, which needs a PRO account
   or a free account Hugging Face counts as eligible.

    python scripts/deploy_hf.py --skip-models    # redeploy only the Space, with the current model commit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parent.parent
FILES = [f"{m}/{f}" for m in ("lm", "parser") for f in ("model.safetensors", "config.json", "tokenizer.json")]
WEB = ["index.html", "style.css", "app.js", "worker.js", "tinygpt.js"]     # web/config.js is written instead


def stage_models(models: Path, repo: str, out: Path) -> None:
    for name in FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(models / name, out / name)
    (out / "README.md").write_text((models / "README.md").read_text().replace("<user>/tinygpt", repo))


def stage_static(repo: str, revision: str, out: Path) -> None:
    """The page as it is in web/, but loading the weights from the model repo at `revision`."""
    web = ROOT / "web"
    for name in WEB:
        shutil.copyfile(web / name, out / name)
    (out / "config.js").write_text(
        "// Written by scripts/deploy_hf.py: the model repo, at the commit deployed with this page.\n"
        f'export const MODEL_BASE = "https://huggingface.co/{repo}/resolve/{revision}/";\n'
        f'export const MODEL_PAGE = "https://huggingface.co/{repo}";\n')
    readme = (web / "SPACE_README.md").read_text()
    for key, value in {"{repo}": repo, "{revision}": revision[:12]}.items():
        readme = readme.replace(key, value)
    (out / "README.md").write_text(readme)


def stage_zerogpu(repo: str, revision: str, out: Path) -> None:
    """The Gradio app with a copy of src/tinygpt; the Space downloads the weights while it is built."""
    import gradio   # only this kind needs it: the Space pins the Gradio version tested here

    space = ROOT / "space"
    shutil.copyfile(space / "app.py", out / "app.py")
    shutil.copyfile(space / "requirements.txt", out / "requirements.txt")
    readme = (space / "README.md").read_text()
    for key, value in {"{gradio_version}": gradio.__version__, "{repo}": repo, "{files}": ",".join(FILES),
                       "{revision}": revision}.items():
        readme = readme.replace(key, value)
    (out / "README.md").write_text(readme)
    (out / "model.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2) + "\n")
    shutil.copytree(ROOT / "src" / "tinygpt", out / "tinygpt", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="tinygpt", help="name of both the model repo and the Space")
    ap.add_argument("--models", default="models", help="what scripts/export.py wrote")
    ap.add_argument("--kind", choices=["static", "zerogpu"], default="static",
                    help="static: the page runs the models in the visitor's browser (free); "
                         "zerogpu: the Gradio app on ZeroGPU (PRO or an eligible account)")
    ap.add_argument("--hardware", default="zero-a10g", help="--kind zerogpu: the Space's hardware")
    ap.add_argument("--skip-models", action="store_true", help="redeploy the Space with the current model commit")
    args = ap.parse_args()

    models = Path(args.models)
    missing = [f for f in [*FILES, "README.md"] if not (models / f).exists()]
    if missing and not args.skip_models:
        sys.exit(f"{models}/ is incomplete ({', '.join(missing)}): run python scripts/export.py first")
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("not logged in: run `hf auth login` with a token that has write access")
    repo = f"{user}/{args.name}"

    if args.skip_models:
        revision = api.model_info(repo).sha
    else:
        api.create_repo(repo, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage_models(models, repo, Path(tmp))
            revision = api.upload_folder(repo_id=repo, folder_path=tmp, commit_message="TinyGPT models").oid
    print(f"model repo: https://huggingface.co/{repo}  (commit {revision[:12]})")

    if args.kind == "static":
        api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage_static(repo, revision, Path(tmp))
            api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp, delete_patterns=["*"],
                              commit_message=f"Deploy the TinyGPT page with model commit {revision[:12]}")
        print(f"space: https://huggingface.co/spaces/{repo}  (live within a minute)")
        return

    try:   # the hardware must be chosen at creation: free accounts may not create Gradio Spaces on CPU
        api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware=args.hardware, exist_ok=True)
    except HfHubHTTPError as e:
        sys.exit(f"could not create the Space: {e}\nZeroGPU needs a PRO account or a free account Hugging Face "
                 "counts as eligible; `--kind static` works on every account.")
    with tempfile.TemporaryDirectory() as tmp:
        stage_zerogpu(repo, revision, Path(tmp))
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=tmp, delete_patterns=["tinygpt/**"],
                          commit_message=f"Deploy TinyGPT with model commit {revision[:12]}")
    try:
        api.request_space_hardware(repo, args.hardware)
    except HfHubHTTPError as e:
        print(f"! could not switch the Space to {args.hardware}: {e}\n"
              f"  Choose the hardware in the Space's Settings: https://huggingface.co/spaces/{repo}/settings")
    print(f"space: https://huggingface.co/spaces/{repo}  (building; the first start takes a few minutes)")


if __name__ == "__main__":
    main()
