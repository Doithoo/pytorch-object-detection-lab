import torch

from object_detector.data.dataset import VocDetectionDataset, detection_collate
from tests.conftest import PreparedVoc


def test_dataset_filters_difficult_only_for_training(prepared_voc: PreparedVoc) -> None:
    train = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=True)
    evaluate = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=False)

    train_image, train_target = train[1]
    _, eval_target = evaluate[1]

    assert train_image.shape == (3, 10, 20)
    assert train_image.dtype == torch.float32
    assert train_target["difficult"].sum().item() == 0
    assert train_target["boxes"].shape == (1, 4)
    assert eval_target["difficult"].sum().item() == 1
    assert eval_target["iscrowd"].tolist() == [0, 1]


def test_empty_target_and_collate_keep_variable_shapes(prepared_voc: PreparedVoc) -> None:
    valid = VocDetectionDataset.from_manifests(prepared_voc.manifests, "valid", training=False)
    train = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=True)

    empty = valid[0]
    images, targets = detection_collate([train[0], empty])

    assert len(images) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["labels"].shape == (0,)
