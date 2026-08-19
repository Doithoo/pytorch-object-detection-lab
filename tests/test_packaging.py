from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COPY_IGNORES = shutil.ignore_patterns(
    ".git",
    ".venv",
    "build",
    "dist",
    "*.egg-info",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)


def _read_metadata(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    return tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))


def _repository_files(project_root: Path, directory: str, suffixes: set[str]) -> set[str]:
    root = project_root / directory
    return {
        path.relative_to(project_root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }


def _read_sdist_files(archive_path: Path) -> set[str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        file_paths = [PurePosixPath(member.name) for member in archive.getmembers() if member.isfile()]

    roots = {path.parts[0] for path in file_paths}
    assert len(roots) == 1, f"sdist must have one root directory, found: {sorted(roots)}"
    return {PurePosixPath(*path.parts[1:]).as_posix() for path in file_paths}


def _read_wheel_files(archive_path: Path) -> set[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _assert_exact_members(actual: set[str], expected: set[str], archive_name: str) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    assert actual == expected, (
        f"{archive_name} member mismatch\nmissing: {missing or 'none'}\nunexpected: {unexpected or 'none'}"
    )


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    workspace = tmp_path_factory.mktemp("packaging")
    project_copy = workspace / "project"
    output_dir = workspace / "dist"
    shutil.copytree(PROJECT_ROOT, project_copy, ignore=COPY_IGNORES)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output_dir)],
        cwd=project_copy,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"offline distribution build failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    sdists = sorted(output_dir.glob("*.tar.gz"))
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(sdists) == 1, f"expected one sdist, found: {sdists}"
    assert len(wheels) == 1, f"expected one wheel, found: {wheels}"
    return project_copy, sdists[0], wheels[0]


def test_version_and_console_name() -> None:
    from object_detector import __version__
    from object_detector.cli import build_parser

    assert __version__ == "0.1.0"
    parser = build_parser()
    assert parser.prog == "detect"


def test_publication_metadata_and_configs_are_declared() -> None:
    metadata = _read_metadata()

    tool = metadata["tool"]
    assert isinstance(tool, dict)
    setuptools = tool["setuptools"]
    assert isinstance(setuptools, dict)
    assert setuptools["packages"]["find"] == {"where": ["src"]}
    assert setuptools["package-data"] == {"object_detector": ["py.typed"]}

    project = metadata["project"]
    assert isinstance(project, dict)
    assert project["urls"]["Source"] == "https://github.com/Yashowhoo/pytorch-object-detection-lab"
    dev_dependencies = project["optional-dependencies"]["dev"]
    assert "tomli>=2,<3; python_version < '3.11'" in dev_dependencies
    assert "setuptools>=77,<82" in dev_dependencies
    assert setuptools["data-files"] == {
        "share/object-detector/configs": [
            "configs/fasterrcnn_resnet50_fpn.yaml",
            "configs/learning_minimal.yaml",
            "configs/reference_fasterrcnn.yaml",
            "configs/ssdlite320_mobilenet_v3.yaml",
        ]
    }


def test_publication_metadata_describes_the_learning_project() -> None:
    metadata = _read_metadata()["project"]
    assert isinstance(metadata, dict)

    assert {"pytorch", "object-detection", "computer-vision", "education"} <= set(metadata["keywords"])
    assert "Intended Audience :: Education" in metadata["classifiers"]
    assert metadata["urls"]["Repository"].endswith("/pytorch-object-detection-lab")
    assert metadata["urls"]["Documentation"].endswith("/pytorch-object-detection-lab#readme")


def test_source_distribution_contains_learning_resources(
    built_distributions: tuple[Path, Path, Path],
) -> None:
    project_copy, sdist_path, _ = built_distributions
    actual = _read_sdist_files(sdist_path)

    expected = {
        "README.md",
        "README.zh-CN.md",
        "CONTRIBUTING.md",
        "CONTRIBUTING.zh-CN.md",
        "SECURITY.md",
        "configs/README.md",
        "configs/README.zh-CN.md",
        "docs/assets/detection-error-analysis.png",
        "docs/assets/detection-target-anatomy.png",
    }
    expected |= _repository_files(project_copy, "configs", {".yaml"})
    expected |= _repository_files(project_copy, "docs", {".md"})
    expected |= _repository_files(
        project_copy,
        "docs/recorded-run",
        {".csv", ".json", ".png", ".py", ".yaml"},
    )
    for directory in ("examples", "scripts", "tests"):
        expected |= _repository_files(project_copy, directory, {".md", ".py"})

    missing = sorted(expected - actual)
    assert not missing, f"sdist is missing learning resources: {missing}"
    github_members = sorted(path for path in actual if path == ".github" or path.startswith(".github/"))
    assert not github_members, f"sdist must not contain GitHub-only files: {github_members}"


def test_wheel_contains_only_runtime_package_and_declared_configs(
    built_distributions: tuple[Path, Path, Path],
) -> None:
    project_copy, _, wheel_path = built_distributions
    metadata = _read_metadata(project_copy)
    project = metadata["project"]
    tool = metadata["tool"]
    assert isinstance(project, dict)
    assert isinstance(tool, dict)
    setuptools = tool["setuptools"]
    assert isinstance(setuptools, dict)

    distribution_name = str(project["name"]).replace("-", "_")
    version = str(project["version"])
    dist_info = f"{distribution_name}-{version}.dist-info"
    data_root = f"{distribution_name}-{version}.data/data"

    expected = _repository_files(project_copy / "src", "object_detector", {".py"})
    expected.add("object_detector/py.typed")
    data_files = setuptools["data-files"]
    assert isinstance(data_files, dict)
    for destination, source_paths in data_files.items():
        for source_path in source_paths:
            expected.add(f"{data_root}/{destination}/{Path(source_path).name}")
    expected |= {
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/RECORD",
        f"{dist_info}/top_level.txt",
    }

    _assert_exact_members(_read_wheel_files(wheel_path), expected, "wheel")


def test_exact_archive_contract_reports_missing_and_leaked_members() -> None:
    with pytest.raises(AssertionError) as error:
        _assert_exact_members(
            {"object_detector/leaked.txt"},
            {"object_detector/expected.py"},
            "fixture wheel",
        )

    message = str(error.value)
    assert "missing: ['object_detector/expected.py']" in message
    assert "unexpected: ['object_detector/leaked.txt']" in message


def test_community_templates_are_present() -> None:
    required = {
        Path(".github/ISSUE_TEMPLATE/bug.yml"),
        Path(".github/ISSUE_TEMPLATE/learning-question.yml"),
        Path(".github/pull_request_template.md"),
        Path("SECURITY.md"),
    }
    assert all(path.is_file() for path in required)
