# PyTorch Object Detection Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publishable, beginner-oriented Pascal VOC 2007 object-detection lab with an offline Faster R-CNN dry run, reproducible training, COCO-style evaluation, error visualization, and checkpoint-only prediction.

**Architecture:** Create an independent `src`-layout package whose typed configuration, VOC data provider, torchvision model registry, training engine, evaluator, and predictor communicate through explicit schemas. Keep torchvision responsible for detector internals and `torchmetrics`/`pycocotools` responsible for AP; keep orchestration and artifacts readable and testable in this repository.

**Tech Stack:** Python 3.10-3.12, PyTorch, torchvision, torchmetrics, pycocotools, Pillow, NumPy, PyYAML, pytest, Ruff, mypy, uv, setuptools, GitHub Actions.

---

## File Map

Create the following focused units. Do not merge them into a single application
module.

```text
pyproject.toml                         package metadata, dependencies, tools, CLI
configs/*.yaml                        learning, reference, and comparison recipes
scripts/download_data.py              verified VOC archive download/extraction
scripts/preview_dataset.py            pre-training image/box inspection
scripts/plot_metrics.py               training-curve rendering
src/object_detector/__init__.py        package version
src/object_detector/cli.py             argparse adapters only
src/object_detector/config.py          typed config and precedence resolution
src/object_detector/preflight.py       aggregated run validation
src/object_detector/data/schema.py     annotation and metadata dataclasses
src/object_detector/data/voc.py        VOC XML parsing and coordinate conversion
src/object_detector/data/manifest.py   official split validation and atomic output
src/object_detector/data/dataset.py    manifest-backed torchvision dataset
src/object_detector/data/transforms.py synchronized image/target transforms
src/object_detector/data/inspection.py preview rendering
src/object_detector/models/spec.py     stable model specification
src/object_detector/models/registry.py model lookup and construction
src/object_detector/models/torchvision_models.py torchvision adapters
src/object_detector/training/trainer.py one-epoch and dry-run mechanics
src/object_detector/training/checkpoint.py atomic checkpoint and resume identity
src/object_detector/training/train.py  run orchestration and artifact lifecycle
src/object_detector/evaluation/metrics.py stable COCO-style metric adapter
src/object_detector/evaluation/errors.py deterministic error matching/ranking
src/object_detector/evaluation/visualization.py annotated output rendering
src/object_detector/evaluation/evaluate.py evaluation orchestration
src/object_detector/inference/predictor.py checkpoint-only single/batch prediction
tests/fixtures/voc.py                  reusable synthetic VOC tree
tests/test_*.py                        unit, integration, CLI, packaging checks
docs/                                  bilingual tutorial, concepts, guides, reference
examples/                              five small executable learning examples
.github/workflows/ci.yml               offline checks and clean-wheel smoke
```

### Task 1: Bootstrap The Installable Package

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `.pre-commit-config.yaml`
- Create: `Makefile`
- Create: `src/object_detector/__init__.py`
- Create: `src/object_detector/cli.py`
- Create: `src/object_detector/py.typed`
- Create: `tests/test_packaging.py`

- [ ] **Step 1: Add package and tool metadata**

Create `pyproject.toml` with package name `object-detector`, version `0.1.0`,
Python `>=3.10,<3.13`, the `detect = "object_detector.cli:main"` entry point,
and these bounded dependencies:

```toml
[build-system]
requires = ["setuptools>=77,<82"]
build-backend = "setuptools.build_meta"

[project]
name = "object-detector"
version = "0.1.0"
description = "Beginner-oriented reproducible PyTorch object detection"
readme = "README.md"
requires-python = ">=3.10,<3.13"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Yashowhoo" }]
dependencies = [
  "torch>=2.0,<3",
  "torchvision>=0.15,<1",
  "torchmetrics>=1.4,<2",
  "pycocotools>=2.0.7,<3",
  "numpy>=1.24,<3",
  "pillow>=9,<12",
  "pyyaml>=6,<7",
]

[project.optional-dependencies]
dev = [
  "pytest>=7,<9", "ruff>=0.6,<1", "mypy>=1.15,<2",
  "types-PyYAML>=6,<7", "pre-commit>=3,<5",
  "build>=1.2,<2", "twine>=5,<7", "matplotlib>=3.6,<4",
]

[project.scripts]
detect = "object_detector.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
object_detector = ["py.typed"]

[tool.ruff]
line-length = 120
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.mypy]
python_version = "3.12"
files = ["src", "scripts", "examples"]
check_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = [
  "torchvision", "torchvision.*", "torchmetrics", "torchmetrics.*",
  "pycocotools", "pycocotools.*",
]
ignore_missing_imports = true
```

Add ignores for `.venv/`, `dist/`, `build/`, `*.egg-info/`, caches,
`data/raw/`, and `artifacts/`. Configure pre-commit with Ruff check and format.
Add Make targets `lint`, `format-check`, `typecheck`, `test`, and `build` that
run through `uv run`.

Create an initial README containing the project name, one-sentence VOC 2007
learning goal, and a link to the approved design. Add the complete MIT license
text with copyright holder `Yashowhoo`, so editable install and package metadata
never reference missing files. Task 14 expands the README into the publication
entry point.

- [ ] **Step 2: Write the failing package and CLI test**

```python
# tests/test_packaging.py
from object_detector import __version__
from object_detector.cli import build_parser


def test_version_and_console_name() -> None:
    assert __version__ == "0.1.0"
    parser = build_parser()
    assert parser.prog == "detect"
```

- [ ] **Step 3: Run the focused test and verify the failure**

Run: `uv sync --extra dev && uv run pytest tests/test_packaging.py -v`

Expected: FAIL during import because `object_detector` has not been implemented.

- [ ] **Step 4: Implement the minimal package and parser**

```python
# src/object_detector/__init__.py
__version__ = "0.1.0"
```

```python
# src/object_detector/cli.py
from __future__ import annotations

import argparse

from object_detector import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
```

- [ ] **Step 5: Verify the package baseline**

Run: `uv run pytest tests/test_packaging.py -v && uv run detect --version`

Expected: one test passes and the CLI prints `0.1.0`.

- [ ] **Step 6: Commit the bootstrap**

```bash
git add pyproject.toml uv.lock README.md LICENSE .gitignore .pre-commit-config.yaml Makefile src tests
git commit -m "build: Scaffold object detection package"
```

### Task 2: Add Typed Configuration And `show-config`

**Files:**
- Create: `src/object_detector/config.py`
- Create: `configs/learning_minimal.yaml`
- Create: `configs/reference_fasterrcnn.yaml`
- Create: `configs/fasterrcnn_resnet50_fpn.yaml`
- Create: `configs/ssdlite320_mobilenet_v3.yaml`
- Create: `tests/test_config.py`
- Modify: `src/object_detector/cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing precedence and validation tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from object_detector.config import ConfigError, load_config


def test_yaml_then_cli_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("train:\n  epochs: 3\n", encoding="utf-8")
    config = load_config(path, [("train.epochs", "5"), ("data.num_workers", "0")])
    assert config.train.epochs == 5
    assert config.data.num_workers == 0
    assert config.model.name == "fasterrcnn_mobilenet_v3_large_320_fpn"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochz: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="train.epochz"):
        load_config(path)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL because `object_detector.config` does not exist.

- [ ] **Step 3: Implement the typed schema and resolver**

Define these frozen dataclasses and public API in `config.py`:

```python
@dataclass(frozen=True)
class DataConfig:
    name: str = "voc2007"
    data_dir: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    num_workers: int = 0
    horizontal_flip: float = 0.5
    max_train_samples: int | None = None
    max_valid_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    weights: str = "none"
    expected_num_classes: int = 21
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    optimizer: str = "sgd"
    scheduler: str = "none"
    seed: int = 42
    amp: bool = False
    grad_clip: float = 0.0
    best_metric: str = "map_50_95"


@dataclass(frozen=True)
class EvaluationConfig:
    score_threshold: float = 0.05
    error_score_threshold: float = 0.5
    error_iou_threshold: float = 0.5
    max_detections: int = 100


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: str = "auto"
    output_dir: Path = Path("artifacts")
    run_name: str | None = None


def load_config(path: Path | None = None, overrides: Sequence[tuple[str, str]] = ()) -> AppConfig:
    """Resolve defaults, YAML, then dotted KEY VALUE overrides."""


def config_to_dict(config: AppConfig) -> dict[str, object]:
    """Return a YAML-safe dictionary with paths serialized as strings."""
```

Implement recursive known-field validation before dataclass construction. Parse
each override value with `yaml.safe_load`, require a nonempty dotted key and one
value, validate probability ranges and positive counts, and raise `ConfigError`
with the dotted field path.

- [ ] **Step 4: Add the four explicit YAML recipes**

The minimal recipe must set `weights: none`, `num_workers: 0`, bounded sample
counts `32/16/16`, two epochs, batch size two, and AMP off. The reference recipe
must set `weights: imagenet1k_v1` and no sample bounds. Comparison recipes use
the registered ResNet50 Faster R-CNN and SSDLite names without pretrained
weights.

- [ ] **Step 5: Wire `show-config` and package the YAML files**

Add a `show-config` subparser with `--config` and repeatable
`--set KEY VALUE` pairs. Its handler
calls `load_config`, prints `yaml.safe_dump(config_to_dict(config), sort_keys=False)`,
and returns zero. Add all four YAML paths to setuptools data files.

- [ ] **Step 6: Verify behavior and formatting**

Run: `uv run pytest tests/test_config.py tests/test_packaging.py -v`

Run: `uv run detect show-config --config configs/learning_minimal.yaml`

Expected: tests pass and output shows `weights: none`, `epochs: 2`, and
`num_workers: 0`.

- [ ] **Step 7: Commit configuration support**

```bash
git add pyproject.toml configs src/object_detector tests/test_config.py
git commit -m "feat(config): Add typed configuration resolution"
```

### Task 3: Define VOC Schemas And Parse XML Correctly

**Files:**
- Create: `src/object_detector/data/__init__.py`
- Create: `src/object_detector/data/schema.py`
- Create: `src/object_detector/data/voc.py`
- Create: `tests/test_voc.py`

- [ ] **Step 1: Write boundary, difficult, and malformed annotation tests**

```python
# tests/test_voc.py
from pathlib import Path

import pytest

from object_detector.data.voc import VocFormatError, parse_voc_annotation


def test_voc_box_becomes_zero_based_xyxy(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height><depth>3</depth></size><object><name>dog</name>"
        "<difficult>1</difficult><bndbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    annotation = parse_voc_annotation(xml)
    assert annotation.objects[0].box == (0.0, 1.0, 20.0, 10.0)
    assert annotation.objects[0].difficult is True


def test_degenerate_box_is_rejected(tmp_path: Path) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height><depth>3</depth></size><object><name>dog</name>"
        "<bndbox><xmin>8</xmin><ymin>2</ymin><xmax>7</xmax>"
        "<ymax>6</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    with pytest.raises(VocFormatError, match="positive width"):
        parse_voc_annotation(xml)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_voc.py -v`

Expected: FAIL because VOC schema and parser modules do not exist.

- [ ] **Step 3: Implement immutable schemas and the parser**

```python
# src/object_detector/data/schema.py
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


@dataclass(frozen=True)
class VocObject:
    class_name: str
    box: tuple[float, float, float, float]
    difficult: bool


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]
```

Use `xml.etree.ElementTree`, require filename and positive image dimensions,
reject unknown classes and non-finite values, convert
`[xmin, ymin, xmax, ymax]` to `[xmin-1, ymin-1, xmax, ymax]`, clip to
`[0, width] x [0, height]`, and reject non-positive boxes. Include the XML path
and object index in every `VocFormatError`.

- [ ] **Step 4: Run parser tests and static checks**

Run: `uv run pytest tests/test_voc.py -v && uv run ruff check src/object_detector/data tests/test_voc.py`

Expected: all parser tests and Ruff checks pass.

- [ ] **Step 5: Commit the VOC contract**

```bash
git add src/object_detector/data tests/test_voc.py
git commit -m "feat(data): Parse Pascal VOC annotations"
```

### Task 4: Prepare Verified Official Manifests

**Files:**
- Create: `src/object_detector/data/manifest.py`
- Create: `scripts/download_data.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/voc.py`
- Create: `tests/conftest.py`
- Create: `tests/test_manifest.py`
- Create: `tests/test_download_data.py`
- Modify: `src/object_detector/cli.py`

- [ ] **Step 1: Build a reusable synthetic VOC fixture**

Implement `build_voc_tree(root: Path) -> Path` in `tests/fixtures/voc.py`. It
must create `VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets/Main}`, four
small RGB JPEGs, matching XML files, and disjoint `train.txt`, `val.txt`, and
`test.txt`. Include one image with two objects, one difficult object, and one
image with no objects.

During Step 4, after `prepare_voc2007` exists, define the shared pytest fixture
in `tests/conftest.py` with this stable shape:

```python
@dataclass(frozen=True)
class PreparedVoc:
    voc_root: Path
    manifests: Path
    metadata: DatasetMetadata


@pytest.fixture
def prepared_voc(tmp_path: Path) -> PreparedVoc:
    voc_root = build_voc_tree(tmp_path / "raw")
    manifests = tmp_path / "manifests"
    metadata = prepare_voc2007(voc_root.parent.parent, manifests, expected_split_counts=None)
    return PreparedVoc(voc_root, manifests, metadata)
```

- [ ] **Step 2: Write failing manifest identity and atomicity tests**

```python
# tests/test_manifest.py
def test_prepare_uses_official_splits_and_stable_hash(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    first = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests", expected_split_counts=None)
    second = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests-2", expected_split_counts=None)
    assert first.split_counts == {"train": 2, "valid": 1, "test": 1}
    assert first.identity == second.identity
    assert (tmp_path / "manifests" / "dataset.yaml").is_file()


def test_overlap_does_not_replace_existing_manifests(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    output = tmp_path / "manifests"
    prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    original = (output / "dataset.yaml").read_bytes()
    (voc_root / "ImageSets/Main/val.txt").write_text("train-1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="split overlap"):
        prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    assert (output / "dataset.yaml").read_bytes() == original
```

- [ ] **Step 3: Run the manifest tests and verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: FAIL because manifest preparation is not implemented.

- [ ] **Step 4: Implement preparation and atomic directory replacement**

Define:

```python
@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    class_names: tuple[str, ...]
    label_by_name: dict[str, int]
    split_counts: dict[str, int]
    split_hashes: dict[str, str]
    identity: str
    coordinate_convention: str


VOC2007_SPLIT_COUNTS = {"train": 2501, "valid": 2510, "test": 4952}


def prepare_voc2007(
    data_dir: Path,
    manifest_dir: Path,
    expected_split_counts: Mapping[str, int] | None = VOC2007_SPLIT_COUNTS,
) -> DatasetMetadata:
    """Validate VOC2007 official splits and atomically write portable manifests."""
```

CSV columns are `image_id,image_path,annotation_path`. Hash normalized rows with
SHA-256. Serialize paths relative to the VOC2007 directory. Write `train.csv`,
`valid.csv`, `test.csv`, `dataset.yaml`, `source.yaml`, and `summary.txt` into a
sibling temporary directory, then replace the destination only after all files
are complete. Reject missing files, duplicate IDs, split overlap, filename
mismatch, and annotation parse errors.
When expected counts are supplied, reject any split count mismatch. Production
CLI calls keep the official default; only synthetic tests pass `None`.

- [ ] **Step 5: Implement the verified downloader**

Use these immutable archive records:

```python
ARCHIVES = (
    VocArchive(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        "VOCtrainval_06-Nov-2007.tar",
        "VOCtrainval_06-Nov-2007.tar",
        "c52e279531787c972589f7e41ab4ae64",
    ),
    VocArchive(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        "VOCtest_06-Nov-2007.tar",
        "VOCtest_06-Nov-2007.tar",
        "b6e924de25625d8de591ea690078ad9f",
    ),
)
```

Download to `*.part`, stream MD5 verification, rename only after verification,
and reject absolute paths, members that escape the destination, symbolic links,
and hard links before extraction. Unit tests must mock the download response and
include a checksum mismatch, a `../escape` member, and a link member.

- [ ] **Step 6: Add and test `prepare-data` CLI routing**

Add `detect prepare-data --data-dir --manifest-dir`. The handler prints the
dataset identity and the three split counts. Test it against `build_voc_tree`
without using the network.

- [ ] **Step 7: Verify the data-preparation slice**

Run: `uv run pytest tests/test_voc.py tests/test_manifest.py tests/test_download_data.py tests/test_cli.py -v`

Expected: all tests pass and the overlap case preserves the previous manifests.

- [ ] **Step 8: Commit data preparation**

```bash
git add scripts/download_data.py src/object_detector/data src/object_detector/cli.py tests
git commit -m "feat(data): Prepare verified VOC 2007 manifests"
```

### Task 5: Load, Transform, Collate, And Preview Detection Samples

**Files:**
- Create: `src/object_detector/data/dataset.py`
- Create: `src/object_detector/data/transforms.py`
- Create: `src/object_detector/data/inspection.py`
- Create: `scripts/preview_dataset.py`
- Create: `tests/test_dataset.py`
- Create: `tests/test_transforms.py`
- Create: `tests/test_inspection.py`

- [ ] **Step 1: Write failing tensor-contract and flip tests**

```python
def test_dataset_filters_difficult_only_for_training(prepared_voc: PreparedVoc) -> None:
    train = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=True)
    evaluate = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=False)
    _, train_target = train[1]
    _, eval_target = evaluate[1]
    assert train_target["difficult"].sum().item() == 0
    assert eval_target["difficult"].sum().item() == 1
    assert eval_target["iscrowd"].tolist() == [0, 1]


def test_horizontal_flip_updates_xyxy() -> None:
    image = torch.zeros((3, 10, 20))
    target = detection_target(boxes=[[2.0, 1.0, 8.0, 6.0]])
    flipped, result = RandomHorizontalFlip(1.0)(image, target)
    assert flipped.shape == image.shape
    assert result["boxes"].tolist() == [[12.0, 1.0, 18.0, 6.0]]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `uv run pytest tests/test_dataset.py tests/test_transforms.py -v`

Expected: FAIL because dataset and paired transforms do not exist.

- [ ] **Step 3: Implement dataset and transform contracts**

`VocDetectionDataset.__getitem__` loads the manifest row and XML, converts the
PIL image with `torchvision.transforms.functional.pil_to_tensor(...).float()/255`,
and returns the exact target keys `boxes`, `labels`, `image_id`, `area`,
`iscrowd`, and `difficult`. Use empty tensors with shapes `(0, 4)` and `(0,)`.

Implement these public functions/classes:

```python
DetectionTarget = dict[str, torch.Tensor]
DetectionTransform = Callable[[torch.Tensor, DetectionTarget], tuple[torch.Tensor, DetectionTarget]]


class Compose:
    def __call__(self, image: torch.Tensor, target: DetectionTarget) -> tuple[torch.Tensor, DetectionTarget]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def detection_collate(batch: Sequence[tuple[torch.Tensor, DetectionTarget]]) -> tuple[list[torch.Tensor], list[DetectionTarget]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
```

`RandomHorizontalFlip` clones the target, flips boxes using image width, and
preserves label alignment. `ColorJitter` changes only image pixels. A shared
`filter_degenerate_boxes` returns the filtered target and removal count.

- [ ] **Step 4: Implement deterministic preview rendering**

`render_detection_preview(samples, class_names, output, columns=2)` uses Pillow
to draw ordinary boxes as solid lines and difficult boxes as dashed lines, with
class labels on an opaque background. `scripts/preview_dataset.py` accepts a
manifest directory, split, output, and limit and calls this package function.

- [ ] **Step 5: Verify empty, mixed, and preview behavior**

Run: `uv run pytest tests/test_dataset.py tests/test_transforms.py tests/test_inspection.py -v`

Expected: all tests pass; the preview test opens the output with Pillow and
confirms a non-empty RGB image.

- [ ] **Step 6: Commit the data-loading path**

```bash
git add src/object_detector/data scripts/preview_dataset.py tests
git commit -m "feat(data): Load and inspect detection samples"
```

### Task 6: Register Three Torchvision Detectors

**Files:**
- Create: `src/object_detector/models/__init__.py`
- Create: `src/object_detector/models/spec.py`
- Create: `src/object_detector/models/registry.py`
- Create: `src/object_detector/models/torchvision_models.py`
- Create: `tests/test_models.py`
- Create: `tests/test_model_smoke.py`

- [ ] **Step 1: Write failing registry and weight-policy tests**

```python
def test_registry_contains_exact_initial_models() -> None:
    assert set(list_models()) == {
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_resnet50_fpn",
        "ssdlite320_mobilenet_v3_large",
    }


def test_offline_default_disables_all_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> nn.Module:
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(adapters, "fasterrcnn_mobilenet_v3_large_320_fpn", fake_constructor)
    build_model("fasterrcnn_mobilenet_v3_large_320_fpn", 21, "none", {})
    assert captured["weights"] is None
    assert captured["weights_backbone"] is None
```

- [ ] **Step 2: Run registry tests and verify they fail**

Run: `uv run pytest tests/test_models.py -v`

Expected: FAIL because model registry modules do not exist.

- [ ] **Step 3: Implement model specifications and adapters**

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    constructor: Callable[[int, str, Mapping[str, object]], nn.Module]
    family: Literal["two_stage", "one_stage"]
    supported_weights: tuple[str, ...] = ("none", "imagenet1k_v1")
    backbone_weights: Mapping[str, WeightsEnum | None] = field(default_factory=dict)


def build_model(
    name: str,
    num_classes: int,
    weights: str = "none",
    params: Mapping[str, object] | None = None,
) -> nn.Module:
    spec = get_model_spec(name)
    if weights not in spec.supported_weights:
        raise ModelConfigError(f"{name} does not support weights={weights!r}")
    return spec.constructor(num_classes, weights, params or {})


def get_backbone_weight(name: str, policy: str) -> WeightsEnum | None:
    spec = get_model_spec(name)
    if policy not in spec.backbone_weights:
        raise ModelConfigError(f"{name} does not define weights={policy!r}")
    return spec.backbone_weights[policy]
```

For offline construction pass both `weights=None` and `weights_backbone=None`.
For the default reference policy pass
`MobileNet_V3_Large_Weights.IMAGENET1K_V1`. Use the corresponding pinned
ResNet50 and MobileNetV3 enums for comparison adapters. Pass `num_classes`
through public torchvision constructors; never mutate private head attributes.

- [ ] **Step 4: Add a real default-model optimization smoke test**

Construct the default model without weights, use one `320x320` image and one
valid box, call training forward, sum named losses, run backward, and take one
SGD step. Assert all loss values are finite and the loss dictionary is nonempty.

- [ ] **Step 5: Verify registry and real model behavior**

Run: `uv run pytest tests/test_models.py tests/test_model_smoke.py -v`

Expected: all tests pass without a model-weight download.

- [ ] **Step 6: Commit model support**

```bash
git add src/object_detector/models tests/test_models.py tests/test_model_smoke.py
git commit -m "feat(models): Register torchvision detectors"
```

### Task 7: Implement The Readable Training Engine And Dry Run

**Files:**
- Create: `src/object_detector/training/__init__.py`
- Create: `src/object_detector/training/trainer.py`
- Create: `tests/fixtures/models.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Create a deterministic fake detector for fast tests**

```python
class FakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None
            return {"loss_classifier": self.scale.square(), "loss_box_reg": self.scale.abs() * 0.5}
        return [
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]], device=image.device),
                "labels": torch.tensor([1], device=image.device),
                "scores": torch.tensor([0.9], device=image.device),
            }
            for image in images
        ]
```

- [ ] **Step 2: Write failing loss aggregation and dry-run tests**

Test that `train_one_epoch` returns averages for `loss_total`,
`loss_classifier`, and `loss_box_reg`; test that a NaN component raises
`NonFiniteLossError` containing the component and image ID; test that `dry_run`
changes the fake detector parameter and returns image, target, and loss summary.

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `uv run pytest tests/test_trainer.py -v`

Expected: FAIL because trainer functions do not exist.

- [ ] **Step 4: Implement one-epoch and one-batch functions**

```python
@dataclass(frozen=True)
class DryRunResult:
    batch_size: int
    image_shapes: tuple[tuple[int, ...], ...]
    target_counts: tuple[int, ...]
    losses: dict[str, float]


def move_batch(images, targets, device):
    return (
        [image.to(device) for image in images],
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )


def sum_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise TrainingError("detector returned no training losses")
    return torch.stack(tuple(losses.values())).sum()
```

`train_one_epoch` sets train mode, transfers the batch, zeroes gradients, calls
the detector with targets, validates every loss, backpropagates the sum, applies
optional positive gradient clipping, steps the optimizer, and returns
sample-weighted averages. `dry_run` performs exactly one optimizer update and
returns `DryRunResult`.

- [ ] **Step 5: Verify the engine against fake and real detectors**

Run: `uv run pytest tests/test_trainer.py tests/test_model_smoke.py -v`

Expected: all tests pass; the model smoke remains offline.

- [ ] **Step 6: Commit training mechanics**

```bash
git add src/object_detector/training tests/fixtures/models.py tests/test_trainer.py
git commit -m "feat(training): Add detection training engine"
```

### Task 8: Add Atomic Checkpoints, Run Metadata, And Resume Validation

**Files:**
- Create: `src/object_detector/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing round-trip and compatibility tests**

```python
def test_checkpoint_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    save_checkpoint(path, checkpoint_payload())
    loaded = load_checkpoint(path)
    assert loaded["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_manifest_change() -> None:
    checkpoint = checkpoint_payload(manifest_identity="old")
    expected = ResumeIdentity(
        model_name="fake",
        class_names=("background", "dog"),
        manifest_identity="new",
        preprocessing={"resize_owner": "model"},
    )
    with pytest.raises(CheckpointCompatibilityError, match="manifest_identity"):
        validate_resume_identity(checkpoint, expected)
```

- [ ] **Step 2: Run checkpoint tests and verify they fail**

Run: `uv run pytest tests/test_checkpoint.py -v`

Expected: FAIL because checkpoint functions do not exist.

- [ ] **Step 3: Implement the versioned payload and atomic write**

```python
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResumeIdentity:
    model_name: str
    class_names: tuple[str, ...]
    manifest_identity: str
    preprocessing: Mapping[str, object]


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
```

`load_checkpoint` requires schema version one and mapping-shaped content.
`validate_resume_identity` compares model name, ordered classes, manifest
identity, and preprocessing. Define `build_run_metadata` to capture Python,
PyTorch, torchvision, platform, device, seed, and Git revision without failing
outside a Git checkout.

- [ ] **Step 4: Verify checkpoint behavior**

Run: `uv run pytest tests/test_checkpoint.py -v`

Expected: all tests pass, including a monkeypatched `torch.save` failure that
leaves no final or temporary file.

- [ ] **Step 5: Commit artifact identity support**

```bash
git add src/object_detector/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(training): Save reproducible checkpoints"
```

### Task 9: Orchestrate Training, Preflight, Resume, And Artifacts

**Files:**
- Create: `src/object_detector/preflight.py`
- Create: `src/object_detector/training/train.py`
- Create: `src/object_detector/evaluation/__init__.py`
- Create: `src/object_detector/evaluation/metrics.py`
- Create: `tests/test_preflight.py`
- Create: `tests/test_training.py`
- Create: `tests/test_metrics.py`
- Modify: `src/object_detector/cli.py`

- [ ] **Step 1: Write failing preflight aggregation and run tests**

Test that preflight returns every independent error for a missing manifest,
class-count mismatch, unavailable device, and unwritable output. Test a fake
two-epoch run creates `config.yaml`, `run.yaml`, `metrics.csv`, `last.pt`, and
`best.pt`; resume from `last.pt` must append epoch three without duplicating
existing metric rows. Add one perfect-prediction metric test that proves the
maintained backend can return `map_50_95` for the fake validation loop. An
uncached requested backbone weight is a preflight notice, not a validation
error, and must name its cache path and network requirement.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/test_preflight.py tests/test_training.py tests/test_metrics.py -v`

Expected: FAIL because orchestration and preflight are absent.

- [ ] **Step 3: Implement aggregated preflight checks**

```python
@dataclass(frozen=True)
class PreflightIssue:
    field: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]
    notices: tuple[str, ...]


def validate_training_request(config: AppConfig, metadata: DatasetMetadata) -> PreflightReport:
    issues: list[PreflightIssue] = []
    notices: list[str] = []
    required = ("train.csv", "valid.csv", "test.csv", "dataset.yaml")
    for name in required:
        if not (config.data.manifest_dir / name).is_file():
            issues.append(PreflightIssue("data.manifest_dir", f"missing {name}"))
    actual_num_classes = len(metadata.class_names) + 1
    if config.model.expected_num_classes != actual_num_classes:
        issues.append(
            PreflightIssue(
                "model.expected_num_classes",
                f"expected {config.model.expected_num_classes}, dataset requires {actual_num_classes}",
            )
        )
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        issues.append(PreflightIssue("device", "CUDA was requested but is unavailable"))
    if config.device == "mps" and not torch.backends.mps.is_available():
        issues.append(PreflightIssue("device", "MPS was requested but is unavailable"))
    if not is_writable_destination(config.output_dir):
        issues.append(PreflightIssue("output_dir", f"cannot write below {config.output_dir}"))
    if config.model.weights != "none":
        cached = expected_weight_cache_path(config.model.name, config.model.weights)
        if not cached.is_file():
            notices.append(
                f"{config.model.weights} is not cached at {cached}; "
                "model construction requires network access"
            )
    return PreflightReport(tuple(issues), tuple(notices))
```

The implementation must collect independent issues and notices before raising
one `PreflightError` with one line per issue. Print notices before model
construction. For `imagenet1k_v1`, inspect the torchvision hub checkpoint path
and state exactly whether network access is required; never begin a download in
preflight and never fall back to random initialization after a download error.

`is_writable_destination` walks from the requested path to its nearest existing
parent and checks that it is a writable directory. `expected_weight_cache_path`
uses the registered torchvision weight enum URL basename under
`torch.hub.get_dir()/checkpoints`; Task 6 must expose the enum selected for each
model and policy so this check cannot drift from construction.

- [ ] **Step 4: Implement run orchestration with injectable factories**

First add a narrow `DetectionMetric` wrapper around
`MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)`
with `update(predictions, targets)` and `compute() -> dict[str, object]`. Then
define `run_training(config, *, model_factory=build_model,
dataset_factory=VocDetectionDataset.from_manifests,
metric_factory=DetectionMetric) -> RunResult`. Seed Python, NumPy, and torch;
create deterministic bounded datasets; write resolved config and run metadata
atomically; build DataLoaders with `detection_collate`; support `--dry-run`;
evaluate the validation loader through `metric_factory` after every epoch; save
`last.pt` every epoch and `best.pt` only when validation `map_50_95` improves;
append one CSV row per completed epoch.
Construct the detector with `len(metadata.class_names) + 1` classes and save
checkpoint `class_names` as `("background", *metadata.class_names)` so label IDs,
metrics, resume validation, and prediction share one ordered vocabulary.

Resume restores model, optimizer, scheduler, epoch, best value, and history. It
allows only total epochs, device, workers, log level, and output location to
differ after identity validation.

- [ ] **Step 5: Wire `detect train`**

Add `--config`, repeatable `--set KEY VALUE`, `--dry-run`, `--resume`, and `--device`.
Dry-run output must end with `dry-run OK` and print batch size, image shapes,
target counts, and named losses.

- [ ] **Step 6: Verify training and recovery**

Run: `uv run pytest tests/test_preflight.py tests/test_training.py tests/test_checkpoint.py tests/test_trainer.py -v`

Expected: all tests pass and the resume test has exactly three metric rows.

- [ ] **Step 7: Commit orchestration**

```bash
git add src/object_detector/preflight.py src/object_detector/training/train.py src/object_detector/evaluation src/object_detector/cli.py tests
git commit -m "feat(training): Orchestrate reproducible runs"
```

### Task 10: Produce Stable COCO-Style Metrics

**Files:**
- Modify: `src/object_detector/evaluation/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write failing perfect, empty, and difficult metric tests**

Build three two-image cases: perfect predictions, no predictions, and a
prediction matching only an `iscrowd=1` difficult target. Assert the stable
result keys are `map_50_95`, `map_50`, `map_75`, `mar_1`, `mar_10`, `mar_100`,
`per_class`, `image_count`, `target_count`, and `prediction_count`. Empty output
must use numeric zeros, not `-1`, NaN, or missing keys. The difficult-only match
must not increase ordinary target count or become a false positive.

- [ ] **Step 2: Run metric tests and verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`

Expected: the perfect-prediction test from Task 9 passes, while new empty,
difficult, and per-class schema assertions fail against the narrow adapter.

- [ ] **Step 3: Implement the maintained metric adapter**

```python
@dataclass(frozen=True)
class ClassMetrics:
    class_id: int
    class_name: str
    map_50_95: float
    mar_100: float


@dataclass(frozen=True)
class DetectionMetrics:
    map_50_95: float
    map_50: float
    map_75: float
    mar_1: float
    mar_10: float
    mar_100: float
    per_class: tuple[ClassMetrics, ...]
    image_count: int
    target_count: int
    prediction_count: int


class DetectionMetric:
    def __init__(self, class_names: Sequence[str]) -> None:
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        self.class_names = tuple(class_names)
```

Filter background from per-class output, preserve `iscrowd`, pass predictions
and targets unchanged to torchmetrics on CPU, and normalize only backend sentinel
values below zero to zero. Do not round in memory; JSON serialization may use
six decimal places. Record torchmetrics and pycocotools versions alongside
serialized results.

- [ ] **Step 4: Verify metrics**

Run: `uv run pytest tests/test_metrics.py -v`

Expected: all three cases pass and pycocotools emits no network activity.

- [ ] **Step 5: Commit metrics**

```bash
git add src/object_detector/evaluation tests/test_metrics.py
git commit -m "feat(evaluation): Add COCO-style detection metrics"
```

### Task 11: Evaluate, Rank Errors, And Render Evidence

**Files:**
- Create: `src/object_detector/evaluation/errors.py`
- Create: `src/object_detector/evaluation/visualization.py`
- Create: `src/object_detector/evaluation/evaluate.py`
- Create: `tests/test_errors.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_visualization.py`
- Modify: `src/object_detector/cli.py`

- [ ] **Step 1: Write failing deterministic error-matching tests**

Test score-descending greedy matching within the same class. A prediction with
IoU at least `0.5` against an unmatched ordinary target is a match; a prediction
with IoU between zero and `0.5` is `localization`; a prediction whose best IoU
is zero is `false_positive`; each unmatched ordinary target is `missed`.
Predictions matched only to difficult targets are `ignored`. Equal scores use
original prediction index as the stable tie-breaker.

- [ ] **Step 2: Run error tests and verify they fail**

Run: `uv run pytest tests/test_errors.py -v`

Expected: FAIL because error analysis does not exist.

- [ ] **Step 3: Implement matching and serializable records**

```python
@dataclass(frozen=True)
class DetectionError:
    image_id: str
    kind: Literal["localization", "false_positive", "missed", "ignored"]
    class_name: str
    score: float | None
    iou: float
    box: tuple[float, float, float, float]


def analyze_image_errors(
    image_id: str,
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[DetectionError, ...]:
    """Return deterministically ordered errors and ignored difficult matches."""
```

Use `torchvision.ops.box_iou`; do not implement IoU arithmetic again.

- [ ] **Step 4: Implement evaluation orchestration and rendering**

`evaluate_model` switches to eval mode, runs under inference mode, updates
`DetectionMetric`, collects CPU predictions and errors, then atomically writes
`evaluation.json`, `per_class.csv`, `predictions.json`, `errors.csv`, and a
`visualizations/` directory. Render worst missed and false-positive images with
target boxes, predicted boxes, scores, and a small legend. Zero predictions
must still create every tabular artifact and one summary visualization.

- [ ] **Step 5: Wire `detect evaluate`**

Add checkpoint, split, output directory, device, score threshold, and overwrite
arguments. Reconstruct config, model, classes, and manifest identity from the
checkpoint. Refuse a different manifest identity before inference.

- [ ] **Step 6: Verify evaluation outputs**

Run: `uv run pytest tests/test_errors.py tests/test_metrics.py tests/test_visualization.py tests/test_evaluation.py -v`

Expected: all tests pass; the empty-prediction case has all required files and
zero metric values.

- [ ] **Step 7: Commit evaluation and evidence generation**

```bash
git add src/object_detector/evaluation src/object_detector/cli.py tests
git commit -m "feat(evaluation): Report metrics and detection errors"
```

### Task 12: Add Checkpoint-Only Single And Batch Prediction

**Files:**
- Create: `src/object_detector/inference/__init__.py`
- Create: `src/object_detector/inference/predictor.py`
- Create: `tests/test_inference.py`
- Modify: `src/object_detector/cli.py`

- [ ] **Step 1: Write failing checkpoint-only prediction tests**

Create a fake checkpoint and injectable fake model. Assert
`Predictor.from_checkpoint` restores ordered classes without a YAML file. Single
mode writes `<stem>.json` and `<stem>.png`. Directory mode processes only
case-insensitive `.jpg`, `.jpeg`, and `.png` files in sorted relative-path order,
writes `predictions.json`, and reports corrupt images without discarding valid
results. Existing outputs fail unless `overwrite=True`.

- [ ] **Step 2: Run inference tests and verify they fail**

Run: `uv run pytest tests/test_inference.py -v`

Expected: FAIL because predictor modules do not exist.

- [ ] **Step 3: Implement checkpoint reconstruction and prediction**

```python
@dataclass(frozen=True)
class Prediction:
    image: str
    width: int
    height: int
    detections: tuple[PredictedObject, ...]


@dataclass(frozen=True)
class PredictedObject:
    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


class Predictor:
    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: str = "auto",
        model_factory: Callable[..., nn.Module] = build_model,
    ) -> "Predictor":
        checkpoint = load_checkpoint(path)
        model_data = require_mapping(checkpoint, "model")
        class_names = tuple(require_string_sequence(checkpoint, "class_names"))
        params = require_mapping(model_data, "params")
        model = model_factory(
            str(model_data["name"]),
            len(class_names),
            "none",
            params,
        )
        model.load_state_dict(checkpoint["model_state"])
        resolved_device = resolve_device(device)
        model.to(resolved_device).eval()
        return cls(model, class_names, resolved_device)
```

Use the same tensor conversion as the dataset and the same drawing function as
evaluation. Filter by user score threshold after model inference, cap display
only after JSON detections are finalized, and preserve floating-point boxes in
JSON. `require_mapping` and `require_string_sequence` raise
`CheckpointCompatibilityError` with the missing or malformed field path. Always
construct with weight policy `none` during restoration because the checkpoint
state already contains backbone tensors and inference must not download weights.

- [ ] **Step 4: Wire the mutually exclusive prediction CLI**

Add required `--checkpoint`, mutually exclusive required `--image` and
`--input-dir`, required `--output-dir`, plus `--device`, `--score-threshold`,
`--display-limit`, and `--overwrite`.

- [ ] **Step 5: Verify single and batch inference**

Run: `uv run pytest tests/test_inference.py tests/test_checkpoint.py -v`

Expected: all tests pass and corrupt batch input produces a nonempty error list
alongside valid results.

- [ ] **Step 6: Commit prediction**

```bash
git add src/object_detector/inference src/object_detector/cli.py tests/test_inference.py
git commit -m "feat(inference): Predict from self-contained checkpoints"
```

### Task 13: Complete The Offline End-To-End Workflow

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write one synthetic end-to-end acceptance test**

The test must build the synthetic VOC tree, call preparation, load the minimal
config with one-epoch overrides, run training with `FakeDetector`, resume for a
second epoch, evaluate test data, and predict one image from `best.pt`. Assert
the exact artifact set, two metric rows, matching manifest identity in every
artifact, zero network calls, and successful checkpoint-only reconstruction.

- [ ] **Step 2: Add CLI parser/exit-code coverage**

Test help for every subcommand, unknown config fields returning nonzero, invalid
mutually exclusive prediction inputs, `dry-run OK`, and concise errors without
tracebacks for known user-input failures.

- [ ] **Step 3: Run the connected acceptance tests**

Run: `uv run pytest tests/test_end_to_end.py tests/test_cli.py -v`

Expected: PASS because Tasks 1-12 already define the exercised contracts. If a
failure appears, stop this task, add a focused failing test beside the owning
module, repair that module with the smallest change, rerun its focused suite,
and commit the fix before returning to this acceptance test.

- [ ] **Step 4: Run the complete code test suite**

Run: `uv run pytest -v`

Expected: all tests pass with no dataset or model-weight downloads.

- [ ] **Step 5: Commit the connected workflow**

```bash
git add src tests
git commit -m "test: Verify the offline detection workflow"
```

### Task 14: Add Learning Material, Contributor UX, CI, And Release Checks

**Files:**
- Modify: `README.md`
- Create: `README.zh-CN.md`
- Create: `CONTRIBUTING.md`
- Create: `CONTRIBUTING.zh-CN.md`
- Create: `docs/README.md`
- Create: `docs/README.zh-CN.md`
- Create: `docs/tutorial/{README,learning-path,00-basics,01-environment,02-data-and-boxes,03-faster-rcnn,04-training,05-evaluation-and-inference}.md`
- Create: matching `docs/tutorial/*.zh-CN.md` files
- Create: `docs/concepts/{code-tour,detection-flow,how-faster-rcnn-works}.md`
- Create: matching `docs/concepts/*.zh-CN.md` files
- Create: `docs/guides/{using-your-data,experiments,troubleshooting,adding-datasets,adding-models}.md`
- Create: matching `docs/guides/*.zh-CN.md` files
- Create: `docs/reference/{config-reference,dataset-format,model-zoo,metrics,checkpoint-schema}.md`
- Create: matching `docs/reference/*.zh-CN.md` files
- Create: `examples/01_boxes_and_labels.py`
- Create: `examples/02_detection_batch.py`
- Create: `examples/03_detector_losses.py`
- Create: `examples/04_minimal_training_loop.py`
- Create: `examples/05_checkpoint_prediction.py`
- Create: `examples/README.md`
- Create: `examples/README.zh-CN.md`
- Create: `scripts/plot_metrics.py`
- Create: `tests/test_examples.py`
- Create: `tests/test_documentation.py`
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing documentation and example contract tests**

`test_documentation.py` must verify every local Markdown link resolves, English
and Chinese primary pages exist in pairs, no full-VOC metric is claimed, README
contains the exact seven-stage workflow, and every documented `detect` command
uses a real parser option. `test_examples.py` executes all five examples with
local synthetic inputs or checks their `--help` path; none may download data or
weights.

- [ ] **Step 2: Run the contract tests and verify they fail**

Run: `uv run pytest tests/test_documentation.py tests/test_examples.py -v`

Expected: FAIL listing the missing documentation and examples.

- [ ] **Step 3: Write the two README entry points**

Both READMEs must include audience, the exact
`download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`
path, uv installation, verified VOC preparation, offline dry run, bounded train,
evaluation, prediction, artifact explanation, model list, repository map,
development commands, license, and an explicit statement that the reference
configuration has no published full-VOC score. Use only visuals produced by a
real bounded run and identify their source; until those exist, omit the image
entirely.

- [ ] **Step 4: Write tutorials, concepts, guides, and references**

Each tutorial chapter ends with one runnable command and one observable expected
result. The configuration reference documents every dataclass field and
precedence. Dataset reference states the one-based-inclusive to zero-based-xyxy
conversion and difficult-object policy. Metrics reference distinguishes
`map_50_95`, `map_50`, thresholds used for error analysis, and test-set
reservation. Checkpoint reference enumerates schema version one and allowed
resume overrides. English and Chinese files must describe identical behavior.

- [ ] **Step 5: Write five progressive examples and plotting utility**

Examples must progress from tensor box structure, collated variable targets,
training loss dictionary, one fake-detector optimization step, to checkpoint
prediction. Each has a `main()` and accepts local fixture paths where data is
needed. `plot_metrics.py` reads `metrics.csv`, requires epoch and loss columns,
and writes a noninteractive Matplotlib PNG.

- [ ] **Step 6: Add publication and contribution files**

Keep the MIT license created in Task 1 unchanged. Contribution guides explain
uv setup, offline tests, formatting, type checking, commit policy, and the rule
against adding network-dependent tests. Update package URLs and package the four
configs in the wheel.

- [ ] **Step 7: Commit the publication surface**

```bash
git add README.md README.zh-CN.md CONTRIBUTING* docs examples scripts pyproject.toml uv.lock tests
git commit -m "docs: Publish the object detection learning path"
```

- [ ] **Step 8: Add offline matrix CI and clean-wheel smoke**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  checks:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - run: uv sync --locked --extra dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy
      - run: uv run pytest -W error::DeprecationWarning
  package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.12"
      - run: uv sync --locked --extra dev
      - run: uv run python -m build
      - run: uv run twine check dist/*
      - run: uv venv --python 3.12 /tmp/object-detector-wheel-smoke
      - run: uv pip install --python /tmp/object-detector-wheel-smoke/bin/python dist/*.whl
      - run: /tmp/object-detector-wheel-smoke/bin/detect --version
      - run: /tmp/object-detector-wheel-smoke/bin/detect show-config
```

- [ ] **Step 9: Verify documentation, examples, and package**

Run: `uv run pytest tests/test_documentation.py tests/test_examples.py tests/test_packaging.py -v`

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy`

Run: `uv build && uv run twine check dist/*`

Expected: all commands exit zero; the wheel contains `object_detector/py.typed`
and the four configuration files.

- [ ] **Step 10: Commit CI after verification**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: Verify offline builds and packages"
```

### Task 15: Run Final Release Verification

**Files:**
- Modify only files implicated by a failing verification command

- [ ] **Step 1: Verify repository cleanliness before release checks**

Run: `git status --short`

Expected: no output.

- [ ] **Step 2: Run all static and behavioral checks from a locked environment**

Run: `uv sync --locked --extra dev`

Run: `uv run ruff check .`

Run: `uv run ruff format --check .`

Run: `uv run mypy`

Run: `uv run pytest -W error::DeprecationWarning -v`

Expected: every command exits zero; pytest reports no failures, errors, skips
caused by missing network, or unexpected warnings.

- [ ] **Step 3: Verify distributions and isolated installation**

Run: `uv build && uv run twine check dist/*`

Run: `uv venv --python 3.12 /tmp/object-detector-wheel-smoke`

Run: `uv pip install --python /tmp/object-detector-wheel-smoke/bin/python dist/*.whl`

Run: `/tmp/object-detector-wheel-smoke/bin/detect --version`

Run: `/tmp/object-detector-wheel-smoke/bin/detect show-config --config configs/learning_minimal.yaml`

Expected: build and Twine checks succeed, CLI prints `0.1.0`, and the resolved
config shows the offline default model and `weights: none`.

- [ ] **Step 4: Execute the bounded synthetic acceptance workflow**

Run: `uv run pytest tests/test_end_to_end.py::test_offline_workflow -v -s`

Expected: PASS after preparing manifests, printing `dry-run OK`, writing two
training epochs across resume, creating zero-safe evaluation artifacts, and
performing checkpoint-only prediction.

- [ ] **Step 5: Audit the worktree and commit history**

Run: `git status --short`

Run: `git log --oneline --decorate -15`

Expected: no uncommitted files and a sequence of scoped Conventional Commits
matching the tasks above. Do not publish or claim a full VOC reference score.
