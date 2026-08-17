from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import unquote

import pytest

from object_detector.cli import build_parser

ROOT = Path(__file__).parents[1]
WORKFLOW = "download -> prepare -> inspect -> dry run -> train -> evaluate -> predict"
DOC_GROUPS = {
    "tutorial": [
        "README",
        "learning-path",
        "00-basics",
        "01-environment",
        "02-data-and-boxes",
        "03-faster-rcnn",
        "04-training",
        "05-evaluation-and-inference",
    ],
    "concepts": ["code-tour", "detection-flow", "how-faster-rcnn-works"],
    "guides": ["using-your-data", "experiments", "troubleshooting", "adding-datasets", "adding-models"],
    "reference": ["config-reference", "dataset-format", "model-zoo", "metrics", "checkpoint-schema"],
}


def _publication_pages() -> list[Path]:
    pages = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CONTRIBUTING.zh-CN.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "examples" / "README.md",
        ROOT / "examples" / "README.zh-CN.md",
    ]
    for group, names in DOC_GROUPS.items():
        for name in names:
            pages.extend((ROOT / "docs" / group / f"{name}.md", ROOT / "docs" / group / f"{name}.zh-CN.md"))
    return pages


def test_english_and_chinese_publication_pages_exist_in_pairs() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in _publication_pages() if not path.is_file()]
    assert not missing, "missing publication pages:\n" + "\n".join(missing)


def test_all_local_markdown_links_resolve() -> None:
    missing = []
    for source in _publication_pages():
        if not source.is_file():
            continue
        for raw_target in re.findall(r"\[[^]]*]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            target = unquote(raw_target.split()[0]).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "broken local links:\n" + "\n".join(missing)


@pytest.mark.parametrize("readme", [ROOT / "README.md", ROOT / "README.zh-CN.md"])
def test_readmes_state_workflow_and_metric_scope(readme: Path) -> None:
    content = readme.read_text(encoding="utf-8")
    assert WORKFLOW in content
    assert "no published full-VOC score" in content


def test_documented_detect_commands_use_real_parser_options() -> None:
    parser = build_parser()
    failures = []
    for source in _publication_pages():
        if not source.is_file():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            command = line.strip()
            if not command.startswith("detect "):
                continue
            try:
                parser.parse_args(shlex.split(command)[1:])
            except SystemExit as exc:
                failures.append(f"{source.relative_to(ROOT)}: {command} (exit {exc.code})")
    assert not failures, "invalid documented commands:\n" + "\n".join(failures)
