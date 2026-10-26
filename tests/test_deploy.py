"""What scripts/deploy_hf.py uploads, checked without uploading: the static Space's files and the
ZeroGPU Space's files."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("deploy_hf", ROOT / "scripts" / "deploy_hf.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

REPO, REVISION = "someone/tinygpt", "0123456789abcdef0123456789abcdef01234567"


def front_matter(readme: str) -> dict[str, str]:
    head = readme.split("---\n")[1]
    return dict(re.findall(r"^(\w+): (.+)$", head, re.M))


def test_static_space_is_the_page_pointing_at_the_pinned_model_commit(tmp_path):
    deploy.stage_static(REPO, REVISION, tmp_path)
    staged = sorted(p.name for p in tmp_path.iterdir())
    assert staged == sorted([*deploy.WEB, "config.js", "README.md"])
    for name in deploy.WEB:                       # the page itself is uploaded unchanged
        assert (tmp_path / name).read_bytes() == (ROOT / "web" / name).read_bytes()
    config = (tmp_path / "config.js").read_text()
    assert f'MODEL_BASE = "https://huggingface.co/{REPO}/resolve/{REVISION}/"' in config
    assert f'MODEL_PAGE = "https://huggingface.co/{REPO}"' in config
    readme = (tmp_path / "README.md").read_text()
    assert "{" not in readme                      # every placeholder filled
    meta = front_matter(readme)
    assert meta["sdk"] == "static" and meta["app_file"] == "index.html"
    assert len(meta["short_description"]) <= 60
    assert f"  - {REPO}" in readme and REVISION[:12] in readme


def test_every_file_the_page_imports_is_uploaded():
    web = ROOT / "web"
    imported = {m for f in ("app.js", "worker.js") for m in re.findall(r'from "\./([\w.]+)"', (web / f).read_text())}
    imported |= set(re.findall(r'new URL\("\./([\w.]+)"', (web / "app.js").read_text()))
    linked = set(re.findall(r'(?:href|src)="([\w.]+\.(?:css|js))"', (web / "index.html").read_text()))
    assert imported | linked <= {*deploy.WEB, "config.js"}


def test_zerogpu_space_has_the_app_the_code_and_the_pinned_models(tmp_path):
    pytest.importorskip("gradio")
    deploy.stage_zerogpu(REPO, REVISION, tmp_path)
    assert (tmp_path / "app.py").exists() and (tmp_path / "tinygpt" / "serve" / "engine.py").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "{" not in readme and front_matter(readme)["sdk"] == "gradio"
    assert f"{REPO} {','.join(deploy.FILES)} {REVISION}" in readme
