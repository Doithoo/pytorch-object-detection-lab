from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from object_detector.config import AppConfig, load_config
from object_detector.training.checkpoint import (
    CheckpointCompatibilityError,
    build_run_metadata,
    load_checkpoint,
    save_checkpoint,
)
from object_detector.training.train import _restore_rng_state, run_training
from tests.conftest import PreparedVoc
from tests.fixtures.models import FakeDetector


def _config(prepared_voc: PreparedVoc, tmp_path: Path, epochs: int, run_name: str = "test-run") -> AppConfig:
    config = load_config()
    return replace(
        config,
        data=replace(
            config.data,
            data_dir=prepared_voc.voc_root.parent.parent,
            manifest_dir=prepared_voc.manifests,
            horizontal_flip=0.0,
            max_train_samples=2,
            max_valid_samples=1,
            max_test_samples=1,
        ),
        model=replace(
            config.model, name="fake", weights="none", expected_num_classes=len(prepared_voc.metadata.class_names) + 1
        ),
        train=replace(config.train, epochs=epochs, batch_size=1, lr=0.1),
        device="cpu",
        output_dir=tmp_path / "artifacts",
        run_name=run_name,
    )


def _fake_model_factory(*args, **kwargs):
    return FakeDetector()


def test_dry_run_rejects_resume_before_accessing_data_or_model(tmp_path: Path) -> None:
    config = load_config()
    config = replace(
        config,
        data=replace(config.data, manifest_dir=tmp_path / "missing-manifests"),
        output_dir=tmp_path / "artifacts",
        run_name="dry-run",
    )

    def fail_model_factory(*args, **kwargs):
        raise AssertionError("model factory must not be called")

    with pytest.raises(ValueError, match="--dry-run cannot be combined with --resume"):
        run_training(
            config,
            resume=tmp_path / "missing.pt",
            dry_run_mode=True,
            model_factory=fail_model_factory,
        )


def test_training_writes_and_resumes_complete_artifacts(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    first = run_training(_config(prepared_voc, tmp_path, 2), model_factory=_fake_model_factory)

    expected = {"config.yaml", "run.yaml", "metrics.csv", "last.pt", "best.pt"}
    assert expected <= {path.name for path in first.run_dir.iterdir()}
    with (first.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    resumed = run_training(
        _config(prepared_voc, tmp_path, 3),
        resume=first.run_dir / "last.pt",
        model_factory=_fake_model_factory,
    )

    with (resumed.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]


class StochasticDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            sample = torch.rand((), device=self.scale.device)
            return {"loss_classifier": (self.scale - sample).square()}
        return [
            {
                "boxes": torch.empty((0, 4), device=image.device),
                "labels": torch.empty((0,), dtype=torch.int64, device=image.device),
                "scores": torch.empty((0,), device=image.device),
            }
            for image in images
        ]


def test_resumed_training_matches_uninterrupted_rng_stream(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    def factory(*args, **kwargs):
        return StochasticDetector()

    continuous = run_training(_config(prepared_voc, tmp_path, 2, "continuous"), model_factory=factory)
    interrupted = run_training(_config(prepared_voc, tmp_path, 1, "interrupted"), model_factory=factory)
    resumed = run_training(
        _config(prepared_voc, tmp_path, 2, "resumed"),
        resume=interrupted.run_dir / "last.pt",
        model_factory=factory,
    )

    continuous_state = load_checkpoint(continuous.run_dir / "last.pt")["model_state"]
    resumed_state = load_checkpoint(resumed.run_dir / "last.pt")["model_state"]
    assert torch.equal(continuous_state["scale"], resumed_state["scale"])


def test_resume_into_new_run_preserves_the_historical_best_checkpoint(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.2, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    def metric_factory(class_names):
        return FixedMetric()

    source = run_training(
        _config(prepared_voc, tmp_path, 2, "source"),
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )
    source_best = load_checkpoint(source.run_dir / "best.pt")
    source_last = load_checkpoint(source.run_dir / "last.pt")
    assert source_best["epoch"] == 1
    assert source_last["epoch"] == 2
    assert not torch.equal(source_best["model_state"]["scale"], source_last["model_state"]["scale"])

    resumed = run_training(
        _config(prepared_voc, tmp_path, 3, "resumed"),
        resume=source.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )

    resumed_best = load_checkpoint(resumed.run_dir / "best.pt")
    resumed_last = load_checkpoint(resumed.run_dir / "last.pt")
    assert resumed_best["epoch"] == source_best["epoch"]
    assert resumed_best["best_metric"] == source_best["best_metric"]
    assert torch.equal(resumed_best["model_state"]["scale"], source_best["model_state"]["scale"])
    assert not torch.equal(resumed_best["model_state"]["scale"], resumed_last["model_state"]["scale"])


def test_resume_chain_preserves_lineage_and_the_original_historical_best(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.2, 0.1, 0.05))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    def metric_factory(class_names):
        return FixedMetric()

    run_a = run_training(
        _config(prepared_voc, tmp_path, 2, "run-a"),
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )
    original_best = load_checkpoint(run_a.run_dir / "best.pt")
    run_b = run_training(
        _config(prepared_voc, tmp_path, 3, "run-b"),
        resume=run_a.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )
    run_c = run_training(
        _config(prepared_voc, tmp_path, 4, "run-c"),
        resume=run_b.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )

    checkpoints = [
        original_best,
        load_checkpoint(run_a.run_dir / "last.pt"),
        load_checkpoint(run_b.run_dir / "best.pt"),
        load_checkpoint(run_b.run_dir / "last.pt"),
        load_checkpoint(run_c.run_dir / "best.pt"),
        load_checkpoint(run_c.run_dir / "last.pt"),
    ]
    lineage_id = checkpoints[0]["lineage_id"]
    assert isinstance(lineage_id, str) and lineage_id
    assert {checkpoint["lineage_id"] for checkpoint in checkpoints} == {lineage_id}
    assert checkpoints[-2]["epoch"] == original_best["epoch"] == 1
    assert torch.equal(checkpoints[-2]["model_state"]["scale"], original_best["model_state"]["scale"])


def test_same_directory_extension_can_then_resume_across_directories(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.2, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    def metric_factory(class_names):
        return FixedMetric()

    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory, metric_factory=metric_factory)
    source_best = load_checkpoint(source.run_dir / "best.pt")
    extended = run_training(
        replace(config, train=replace(config.train, epochs=2)),
        resume=source.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )
    resumed = run_training(
        _config(prepared_voc, tmp_path, 3, "resumed"),
        resume=extended.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=metric_factory,
    )

    resumed_best = load_checkpoint(resumed.run_dir / "best.pt")
    assert resumed_best["lineage_id"] == source_best["lineage_id"]
    assert resumed_best["epoch"] == 1
    assert torch.equal(resumed_best["model_state"]["scale"], source_best["model_state"]["scale"])


def test_resume_from_best_into_new_run_preserves_the_resume_checkpoint(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    source_best = load_checkpoint(source.run_dir / "best.pt")

    resumed = run_training(
        _config(prepared_voc, tmp_path, 2, "resumed"),
        resume=source.run_dir / "best.pt",
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )

    resumed_best = load_checkpoint(resumed.run_dir / "best.pt")
    assert resumed_best["epoch"] == source_best["epoch"]
    assert torch.equal(resumed_best["model_state"]["scale"], source_best["model_state"]["scale"])


def test_cross_directory_resume_requires_the_source_best_checkpoint(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    (source.run_dir / "best.pt").unlink()

    with pytest.raises(CheckpointCompatibilityError, match="historical best checkpoint"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("model", "source field model"),
        ("weight_policy", "source field weight_policy"),
        ("class_names", "class_names"),
        ("preprocessing", "preprocessing"),
        ("manifest_identity", "manifest_identity"),
        ("split_hashes", "source field split_hashes"),
    ],
)
def test_cross_directory_resume_rejects_tampered_source_best_identity_field(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    if field == "model":
        source_best[field] = {**source_best[field], "params": {"tampered": True}}
    elif field == "weight_policy":
        source_best[field] = "imagenet1k_v1"
    elif field == "class_names":
        source_best[field] = [*source_best[field][:-1], "tampered-class"]
    elif field == "preprocessing":
        source_best[field] = {**source_best[field], "color_space": "BGR"}
    elif field == "manifest_identity":
        source_best[field] = "different-manifest"
    else:
        source_best[field] = {**source_best[field], "train": "different-split"}
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize("missing_field", ["optimizer_state", "rng_state", "metric_history"])
def test_cross_directory_resume_rejects_an_incomplete_source_best_checkpoint(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    missing_field: str,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    del source_best[missing_field]
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match=missing_field):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("model_state", "model_state"),
        ("optimizer_state", "optimizer_state"),
    ],
)
def test_cross_directory_resume_rejects_an_unloadable_source_best_state(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best[field] = {}
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_rejects_an_unrestorable_source_best_rng_state(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["rng_state"]["python"] = []
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match=r"rng_state\.python"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_allows_cuda_best_with_cpu_last(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["run_metadata"]["device"] = "cuda:0"
    source_best["run_metadata"]["cuda_device_count"] = 1
    source_best["rng_state"]["torch_cuda"] = [torch.ones(1, dtype=torch.uint8)]
    save_checkpoint(source_best_path, source_best)

    resumed = run_training(
        _config(prepared_voc, tmp_path, 2, "resumed"),
        resume=source.run_dir / "last.pt",
        model_factory=_fake_model_factory,
    )

    resumed_best = load_checkpoint(resumed.run_dir / "best.pt")
    resumed_last = load_checkpoint(resumed.run_dir / "last.pt")
    assert resumed_best["run_metadata"]["cuda_device_count"] == 1
    assert len(resumed_best["rng_state"]["torch_cuda"]) == 1
    assert resumed_last["run_metadata"]["cuda_device_count"] == 0
    assert resumed_last["rng_state"]["torch_cuda"] == []


def test_cross_directory_resume_rejects_cuda_rng_count_mismatch(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["run_metadata"]["cuda_device_count"] = 1
    assert source_best["rng_state"]["torch_cuda"] == []
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match="cuda_device_count"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_same_directory_resume_rejects_cuda_rng_count_mismatch(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    source_last["run_metadata"]["cuda_device_count"] = 1
    assert source_last["rng_state"]["torch_cuda"] == []
    save_checkpoint(source_last_path, source_last)

    with pytest.raises(ValueError, match="cuda_device_count"):
        run_training(
            replace(config, train=replace(config.train, epochs=2)),
            resume=source_last_path,
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_history", "metric_history"),
        ("broken_history", "epochs must be consecutive"),
        ("nan_best_metric", "best_metric must be a finite number"),
        ("bool_best_metric", "best_metric must be a finite number"),
        ("scheduler_contract", "scheduler_state must be null"),
    ],
)
def test_same_directory_resume_validates_the_complete_checkpoint_before_loading_state(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    if case == "missing_history":
        del source_last["metric_history"]
    elif case == "broken_history":
        source_last["metric_history"][0]["epoch"] = 2
    elif case == "nan_best_metric":
        source_last["best_metric"] = float("nan")
    elif case == "bool_best_metric":
        source_last["best_metric"] = True
    else:
        source_last["scheduler_state"] = {}
    save_checkpoint(source_last_path, source_last)

    with pytest.raises(ValueError, match=message):
        run_training(
            replace(config, train=replace(config.train, epochs=2)),
            resume=source_last_path,
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize("resume_role", ["resume_checkpoint", "historical_best"])
@pytest.mark.parametrize("tensor_case", ["empty", "two_dimensional", "non_uint8"])
def test_resume_rejects_malformed_cuda_rng_tensor(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    resume_role: str,
    tensor_case: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    checkpoint_path = source.run_dir / ("last.pt" if resume_role == "resume_checkpoint" else "best.pt")
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint["run_metadata"]["device"] = "cuda:0"
    checkpoint["run_metadata"]["cuda_device_count"] = 1
    if tensor_case == "empty":
        cuda_state = torch.empty(0, dtype=torch.uint8)
    elif tensor_case == "two_dimensional":
        cuda_state = torch.ones((1, 1), dtype=torch.uint8)
    else:
        cuda_state = torch.ones(1, dtype=torch.int64)
    checkpoint["rng_state"]["torch_cuda"] = [cuda_state]
    save_checkpoint(checkpoint_path, checkpoint)

    resumed_config = replace(
        config,
        train=replace(config.train, epochs=2),
        run_name="source" if resume_role == "resume_checkpoint" else "resumed",
    )
    with pytest.raises(ValueError, match="nonempty one-dimensional byte tensors"):
        run_training(
            resumed_config,
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize("gating_case", ["current_non_cuda", "cuda_unavailable", "saved_list_empty"])
def test_resume_skips_cuda_rng_restore_when_gated_off(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gating_case: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    if gating_case != "saved_list_empty":
        source_last["run_metadata"]["device"] = "cuda:0"
        source_last["run_metadata"]["cuda_device_count"] = 1
        source_last["rng_state"]["torch_cuda"] = [torch.ones(1, dtype=torch.uint8)]
    else:
        assert source_last["rng_state"]["torch_cuda"] == []

    cuda_available = gating_case != "cuda_unavailable"
    current_device = torch.device("cpu" if gating_case == "current_non_cuda" else "cuda:0")
    restore_calls: list[tuple[torch.Tensor, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restore_calls.append((state, device)),
    )

    _restore_rng_state(source_last, torch.Generator(), current_device)

    assert restore_calls == []


@pytest.mark.parametrize(
    ("saved_device", "cuda_device_count", "message"),
    [
        ("cpu", 1, "non-CUDA device"),
        ("mps", 1, "non-CUDA device"),
        ("cuda", 1, "explicit CUDA device index"),
        ("cuda:1", 1, "source CUDA device index"),
    ],
)
def test_resume_rejects_inconsistent_saved_cuda_device_metadata(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    saved_device: str,
    cuda_device_count: int,
    message: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    source_last["run_metadata"]["device"] = saved_device
    source_last["run_metadata"]["cuda_device_count"] = cuda_device_count
    source_last["rng_state"]["torch_cuda"] = [torch.ones(1, dtype=torch.uint8)] * cuda_device_count
    save_checkpoint(source_last_path, source_last)

    with pytest.raises(ValueError, match=message):
        run_training(
            replace(config, train=replace(config.train, epochs=2)),
            resume=source_last_path,
            model_factory=_fake_model_factory,
        )


def test_resume_maps_saved_cuda_device_rng_to_current_cuda_device(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_training(_config(prepared_voc, tmp_path, 1, "source"), model_factory=_fake_model_factory)
    checkpoint = load_checkpoint(source.run_dir / "last.pt")
    checkpoint["run_metadata"]["device"] = "cuda:7"
    checkpoint["run_metadata"]["cuda_device_count"] = 8
    checkpoint["rng_state"]["torch_cuda"] = [torch.tensor([index], dtype=torch.uint8) for index in range(8)]
    restore_calls: list[tuple[torch.Tensor, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restore_calls.append((state, device)),
    )

    _restore_rng_state(checkpoint, torch.Generator(), torch.device("cuda:0"))

    assert len(restore_calls) == 1
    assert torch.equal(restore_calls[0][0], torch.tensor([7], dtype=torch.uint8))
    assert restore_calls[0][1] == torch.device("cuda:0")


def test_resume_uses_the_normalized_implicit_cuda_source_index(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_training(_config(prepared_voc, tmp_path, 1, "source"), model_factory=_fake_model_factory)
    checkpoint = load_checkpoint(source.run_dir / "last.pt")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    checkpoint["run_metadata"] = build_run_metadata(device=torch.device("cuda"), seed=42)
    checkpoint["rng_state"]["torch_cuda"] = [
        torch.tensor([0], dtype=torch.uint8),
        torch.tensor([1], dtype=torch.uint8),
    ]
    restore_calls: list[tuple[torch.Tensor, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restore_calls.append((state, device)),
    )

    _restore_rng_state(checkpoint, torch.Generator(), torch.device("cuda"))

    assert len(restore_calls) == 1
    assert torch.equal(restore_calls[0][0], torch.tensor([1], dtype=torch.uint8))
    assert restore_calls[0][1] == torch.device("cuda:1")


def test_resume_rejects_a_current_cuda_device_index_that_does_not_exist(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_training(_config(prepared_voc, tmp_path, 1, "source"), model_factory=_fake_model_factory)
    checkpoint = load_checkpoint(source.run_dir / "last.pt")
    checkpoint["run_metadata"]["device"] = "cuda:0"
    checkpoint["run_metadata"]["cuda_device_count"] = 1
    checkpoint["rng_state"]["torch_cuda"] = [torch.ones(1, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="current CUDA device index"):
        _restore_rng_state(checkpoint, torch.Generator(), torch.device("cuda:1"))


def test_cross_directory_resume_rejects_an_unloadable_step_scheduler_state(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source_config = _config(prepared_voc, tmp_path, 1, "source")
    source_config = replace(source_config, train=replace(source_config.train, scheduler="step"))
    source = run_training(source_config, model_factory=_fake_model_factory)
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["scheduler_state"] = {}
    save_checkpoint(source_best_path, source_best)

    resumed_config = replace(
        source_config,
        train=replace(source_config.train, epochs=2),
        run_name="resumed",
    )
    with pytest.raises(CheckpointCompatibilityError, match="scheduler_state"):
        run_training(
            resumed_config,
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize("resume_mode", ["same_directory", "cross_directory"])
def test_resume_rejects_step_scheduler_state_incompatible_with_fresh_scheduler(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    resume_mode: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    config = replace(config, train=replace(config.train, scheduler="step"))
    source = run_training(config, model_factory=_fake_model_factory)
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    source_last["scheduler_state"] = {}
    save_checkpoint(source_last_path, source_last)
    if resume_mode == "cross_directory":
        source_best_path = source.run_dir / "best.pt"
        source_best = load_checkpoint(source_best_path)
        source_best["scheduler_state"] = {}
        save_checkpoint(source_best_path, source_best)

    resumed_config = replace(
        config,
        train=replace(config.train, epochs=2),
        run_name="source" if resume_mode == "same_directory" else "resumed",
    )
    with pytest.raises(ValueError, match="resume checkpoint scheduler_state is incompatible"):
        run_training(
            resumed_config,
            resume=source_last_path,
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_rejects_a_best_checkpoint_from_another_run(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source_metric_values = iter((0.8, 0.2))
    foreign_metric_values = iter((0.8,))

    class FixedMetric:
        def __init__(self, values) -> None:
            self.values = values

        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(self.values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 2, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(source_metric_values),
    )
    foreign = run_training(
        _config(prepared_voc, tmp_path, 1, "foreign"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(foreign_metric_values),
    )
    source_last = load_checkpoint(source.run_dir / "last.pt")
    source_last["lineage_id"] = "source-lineage"
    save_checkpoint(source.run_dir / "last.pt", source_last)
    foreign_best = load_checkpoint(foreign.run_dir / "best.pt")
    foreign_best["lineage_id"] = "foreign-lineage"
    foreign_best["config"] = source_last["config"]
    foreign_best["run_metadata"] = source_last["run_metadata"]
    foreign_best["metric_history"] = source_last["metric_history"][:1]
    save_checkpoint(source.run_dir / "best.pt", foreign_best)

    with pytest.raises(CheckpointCompatibilityError, match="lineage_id"):
        run_training(
            _config(prepared_voc, tmp_path, 3, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_rejects_a_best_history_that_is_not_a_resume_prefix(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.2))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 2, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["metric_history"][0]["loss_total"] = 999.0
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match="metric_history must be an exact prefix"):
        run_training(
            _config(prepared_voc, tmp_path, 3, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_rejects_scheduler_state_for_none_scheduler(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["scheduler_state"] = {}
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(CheckpointCompatibilityError, match="scheduler_state must be null"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_from_best_requires_metric_history_consistency(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8,))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 1, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    source_best_path = source.run_dir / "best.pt"
    source_best = load_checkpoint(source_best_path)
    source_best["metric_history"][0]["valid_map_50_95"] = 0.7
    save_checkpoint(source_best_path, source_best)

    with pytest.raises(ValueError, match="resume checkpoint best_metric must equal the maximum"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "resumed"),
            resume=source_best_path,
            model_factory=_fake_model_factory,
        )


def test_cross_directory_resume_rejects_tied_last_epoch_as_historical_best(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.8))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 2, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    forged_best = load_checkpoint(source.run_dir / "last.pt")
    save_checkpoint(source.run_dir / "best.pt", forged_best)

    with pytest.raises(CheckpointCompatibilityError, match="strictly greater"):
        run_training(
            _config(prepared_voc, tmp_path, 3, "resumed"),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


@pytest.mark.parametrize(
    ("suffix_metric", "error"),
    [
        (0.9, "best_metric must equal the maximum"),
        (float("nan"), "history must contain finite numbers"),
    ],
    ids=["higher", "nonfinite"],
)
def test_cross_directory_resume_rejects_invalid_metric_history_after_historical_best(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    suffix_metric: float,
    error: str,
) -> None:
    metric_values = iter((0.8, 0.2, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    source = run_training(
        _config(prepared_voc, tmp_path, 2, "source"),
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    source_last_path = source.run_dir / "last.pt"
    source_last = load_checkpoint(source_last_path)
    source_last["metric_history"][1]["valid_map_50_95"] = suffix_metric
    save_checkpoint(source_last_path, source_last)

    with pytest.raises(ValueError, match=error):
        run_training(
            _config(prepared_voc, tmp_path, 3, "resumed"),
            resume=source_last_path,
            model_factory=_fake_model_factory,
            metric_factory=lambda class_names: FixedMetric(),
        )


@pytest.mark.parametrize("resume_filename", ["last.pt", "best.pt"])
@pytest.mark.parametrize("destination", ["same_directory", "cross_directory"])
def test_every_resume_checkpoint_requires_best_metric_to_equal_the_history_maximum(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    resume_filename: str,
    destination: str,
) -> None:
    metric_values = iter((0.8, 0.2, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    config = _config(prepared_voc, tmp_path, 2, "source")
    source = run_training(
        config,
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    if resume_filename == "best.pt":
        save_checkpoint(source.run_dir / "best.pt", load_checkpoint(source.run_dir / "last.pt"))
    resume_path = source.run_dir / resume_filename
    checkpoint = load_checkpoint(resume_path)
    checkpoint["metric_history"][1]["valid_map_50_95"] = 0.9
    save_checkpoint(resume_path, checkpoint)
    if destination == "same_directory" and resume_filename == "best.pt":
        (source.run_dir / "last.pt").unlink()

    resumed_config = replace(
        config,
        train=replace(config.train, epochs=3),
        run_name="source" if destination == "same_directory" else "resumed",
    )
    with pytest.raises(ValueError, match="resume checkpoint best_metric must equal the maximum"):
        run_training(
            resumed_config,
            resume=resume_path,
            model_factory=_fake_model_factory,
            metric_factory=lambda class_names: FixedMetric(),
        )


def test_resume_checkpoint_history_allows_a_later_tie_for_best_metric(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.8, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    config = _config(prepared_voc, tmp_path, 2, "source")
    source = run_training(
        config,
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )

    resumed = run_training(
        replace(config, train=replace(config.train, epochs=3)),
        resume=source.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )

    assert load_checkpoint(resumed.run_dir / "last.pt")["epoch"] == 3


def test_same_directory_resume_does_not_require_a_sibling_best_checkpoint(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.1, 0.2))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(
        config,
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    (source.run_dir / "best.pt").unlink()

    resumed = run_training(
        replace(config, train=replace(config.train, epochs=2)),
        resume=source.run_dir / "last.pt",
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )

    assert load_checkpoint(resumed.run_dir / "best.pt")["epoch"] == 2


@pytest.mark.parametrize("resume_source", ["best", "older_copy"])
def test_same_directory_resume_rejects_a_checkpoint_older_than_existing_last(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
    resume_source: str,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    if resume_source == "best":
        resume_path = source.run_dir / "best.pt"
    else:
        old_checkpoint = load_checkpoint(source.run_dir / "last.pt")
        run_training(
            replace(config, train=replace(config.train, epochs=2)),
            resume=source.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )
        resume_path = source.run_dir / "older.pt"
        save_checkpoint(resume_path, old_checkpoint)

    with pytest.raises(ValueError, match=r"existing last\.pt.*new empty run directory"):
        run_training(
            replace(config, train=replace(config.train, epochs=3)),
            resume=resume_path,
            model_factory=_fake_model_factory,
        )


def test_same_directory_resume_from_best_is_allowed_when_last_is_missing(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    (source.run_dir / "last.pt").unlink()

    resumed = run_training(
        replace(config, train=replace(config.train, epochs=2)),
        resume=source.run_dir / "best.pt",
        model_factory=_fake_model_factory,
    )

    assert load_checkpoint(resumed.run_dir / "last.pt")["epoch"] == 2


def test_same_directory_resume_rejects_an_arbitrary_checkpoint_when_last_is_missing(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    config = _config(prepared_voc, tmp_path, 1, "source")
    source = run_training(config, model_factory=_fake_model_factory)
    older_path = source.run_dir / "older.pt"
    save_checkpoint(older_path, load_checkpoint(source.run_dir / "last.pt"))
    (source.run_dir / "last.pt").unlink()

    with pytest.raises(ValueError, match=r"only .*best\.pt.*new empty run directory"):
        run_training(
            replace(config, train=replace(config.train, epochs=2)),
            resume=older_path,
            model_factory=_fake_model_factory,
        )


def test_same_directory_resume_from_best_requires_strict_historical_best_history(
    prepared_voc: PreparedVoc,
    tmp_path: Path,
) -> None:
    metric_values = iter((0.8, 0.8, 0.1))

    class FixedMetric:
        def update(self, predictions, targets) -> None:
            pass

        def compute(self) -> dict[str, float]:
            return {"map_50_95": next(metric_values)}

    config = _config(prepared_voc, tmp_path, 2, "source")
    source = run_training(
        config,
        model_factory=_fake_model_factory,
        metric_factory=lambda class_names: FixedMetric(),
    )
    save_checkpoint(source.run_dir / "best.pt", load_checkpoint(source.run_dir / "last.pt"))
    (source.run_dir / "last.pt").unlink()

    with pytest.raises(ValueError, match="strictly greater"):
        run_training(
            replace(config, train=replace(config.train, epochs=3)),
            resume=source.run_dir / "best.pt",
            model_factory=_fake_model_factory,
            metric_factory=lambda class_names: FixedMetric(),
        )


def test_resume_requires_extending_the_saved_epoch(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    first = run_training(_config(prepared_voc, tmp_path, 1), model_factory=_fake_model_factory)

    with pytest.raises(ValueError, match="must be greater than checkpoint epoch 1"):
        run_training(
            _config(prepared_voc, tmp_path, 1),
            resume=first.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )


def test_resume_rejects_an_unrelated_nonempty_run_directory(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    first = run_training(_config(prepared_voc, tmp_path, 1, "source"), model_factory=_fake_model_factory)
    destination = tmp_path / "artifacts" / "occupied"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("existing experiment", encoding="utf-8")

    with pytest.raises(FileExistsError, match="resume run directory is not the checkpoint directory"):
        run_training(
            _config(prepared_voc, tmp_path, 2, "occupied"),
            resume=first.run_dir / "last.pt",
            model_factory=_fake_model_factory,
        )

    assert {path.name for path in destination.iterdir()} == {marker.name}
