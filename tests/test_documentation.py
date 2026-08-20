from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from urllib.parse import unquote

import pytest

from object_detector.cli import build_parser
from object_detector.models.registry import list_models

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
    "concepts": ["code-tour", "configuration-flow", "detection-flow", "how-faster-rcnn-works"],
    "guides": [
        "using-models",
        "using-your-data",
        "experiments",
        "kaggle",
        "troubleshooting",
        "adding-datasets",
        "adding-models",
    ],
    "reference": ["config-reference", "dataset-format", "model-zoo", "metrics", "checkpoint-schema", "voc2007"],
    "architecture": ["0001-reproducible-voc-detection-contracts"],
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
        ROOT / "configs" / "README.md",
        ROOT / "configs" / "README.zh-CN.md",
        ROOT / "scripts" / "README.md",
        ROOT / "scripts" / "README.zh-CN.md",
        ROOT / "tests" / "README.md",
        ROOT / "tests" / "README.zh-CN.md",
        ROOT / "docs" / "recorded-run" / "README.md",
        ROOT / "docs" / "recorded-run" / "README.zh-CN.md",
    ]
    for group, names in DOC_GROUPS.items():
        for name in names:
            pages.extend((ROOT / "docs" / group / f"{name}.md", ROOT / "docs" / group / f"{name}.zh-CN.md"))
    return pages


def _missing_publication_pages(pages: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in pages if not path.is_file()]


def _broken_local_links(pages: list[Path], root: Path) -> list[str]:
    missing = []
    for source in pages:
        if not source.is_file():
            continue
        for raw_target in re.findall(r"\[[^]]*]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            target = unquote(raw_target.split()[0]).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{source.relative_to(root).as_posix()} -> {target}")
    return missing


@pytest.mark.parametrize("missing_name", ["guide.md", "guide.zh-CN.md"])
def test_missing_publication_pages_reports_absent_language_page(tmp_path: Path, missing_name: str) -> None:
    pages = [tmp_path / "guide.md", tmp_path / "guide.zh-CN.md"]
    for page in pages:
        if page.name != missing_name:
            page.write_text("published\n", encoding="utf-8")

    assert _missing_publication_pages(pages, tmp_path) == [missing_name]


@pytest.mark.parametrize(
    ("markdown", "existing_targets", "expected"),
    [
        ("[missing guide](missing.md)", [], ["README.md -> missing.md"]),
        ("![missing diagram](images/missing.png)", [], ["README.md -> images/missing.png"]),
        ("[guide section](guide.md#details)", ["guide.md"], []),
        ("[web](https://example.com/guide)", [], []),
        ("[web](http://example.com/guide)", [], []),
        ("[email](mailto:maintainer@example.com)", [], []),
        ("[section](#details)", [], []),
    ],
)
def test_broken_local_links_reports_only_unresolved_local_targets(
    tmp_path: Path,
    markdown: str,
    existing_targets: list[str],
    expected: list[str],
) -> None:
    source = tmp_path / "README.md"
    source.write_text(markdown, encoding="utf-8")
    for target in existing_targets:
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Details\n", encoding="utf-8")

    assert _broken_local_links([source], tmp_path) == expected


def test_english_and_chinese_publication_pages_exist_in_pairs() -> None:
    missing = _missing_publication_pages(_publication_pages(), ROOT)
    assert not missing, "missing publication pages:\n" + "\n".join(missing)


def test_all_local_markdown_links_resolve() -> None:
    missing = _broken_local_links(_publication_pages(), ROOT)
    assert not missing, "broken local links:\n" + "\n".join(missing)


@pytest.mark.parametrize("readme", [ROOT / "README.md", ROOT / "README.zh-CN.md"])
def test_readmes_state_workflow_and_metric_scope(readme: Path) -> None:
    content = readme.read_text(encoding="utf-8")
    assert WORKFLOW in content
    assert "0.322312" in content
    assert "docs/recorded-run/" in content


def test_publication_pages_do_not_expose_numbered_kaggle_revisions() -> None:
    historical_url = "https://www.kaggle.com/code/yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7"
    failures = []
    patterns = (r"(?i)\bkaggle\s+v\d+\b", r"(?i)\bgpu-run-v\d+\b")
    for source in _publication_pages():
        content = source.read_text(encoding="utf-8").replace(historical_url, "")
        for pattern in patterns:
            if match := re.search(pattern, content):
                failures.append(f"{source.relative_to(ROOT)}: {match.group(0)}")

    assert not failures, "numbered Kaggle revisions in published docs:\n" + "\n".join(failures)


def test_kaggle_metadata_uses_stable_user_facing_identifier() -> None:
    path = ROOT / "docs" / "recorded-run" / "kaggle" / "kernel-metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["id"] == "yashowhoo/pytorch-object-detection-lab-voc2007-gpu"
    assert metadata["title"] == "PyTorch Object Detection Lab VOC2007 GPU"


def test_documented_detect_commands_use_real_parser_options() -> None:
    parser = build_parser()
    failures = []
    for source in _publication_pages():
        if not source.is_file():
            continue
        content = source.read_text(encoding="utf-8")
        snippets = re.findall(r"`((?:uv run )?detect [^`\n]+)`", content)
        snippets.extend(
            line.strip() for line in content.splitlines() if line.strip().startswith(("detect ", "uv run detect "))
        )
        for command in snippets:
            if command.endswith("\\"):
                continue
            normalized = command.removeprefix("uv run ")
            try:
                parser.parse_args(shlex.split(normalized)[1:])
            except SystemExit as exc:
                if exc.code != 0:
                    failures.append(f"{source.relative_to(ROOT)}: {command} (exit {exc.code})")
    assert not failures, "invalid documented commands:\n" + "\n".join(failures)


def test_documented_python_entry_points_exist() -> None:
    missing = []
    for source in _publication_pages():
        if not source.is_file():
            continue
        content = source.read_text(encoding="utf-8")
        for target in re.findall(r"uv run python\s+([^\s`\\]+)", content):
            if target.startswith("-"):
                continue
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "missing documented Python entry points:\n" + "\n".join(missing)


def test_documented_config_paths_exist() -> None:
    missing = []
    for source in _publication_pages():
        if not source.is_file():
            continue
        content = source.read_text(encoding="utf-8")
        for target in re.findall(r"--config\s+([^\s`\\]+)", content):
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "missing documented config files:\n" + "\n".join(missing)


def test_model_zoo_lists_every_registered_model() -> None:
    for language_suffix in ("", ".zh-CN"):
        path = ROOT / "docs" / "reference" / f"model-zoo{language_suffix}.md"
        content = path.read_text(encoding="utf-8")
        documented = {name for name in list_models() if re.search(rf"\|\s*`{re.escape(name)}`\s*\|", content)}
        assert documented == set(list_models())


def test_generated_documentation_assets_are_nonempty_pngs() -> None:
    for name in ("detection-target-anatomy.png", "detection-error-analysis.png"):
        path = ROOT / "docs" / "assets" / name
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1_000
