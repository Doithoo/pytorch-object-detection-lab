from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch.nn as nn

ModelConstructor = Callable[[int, str, Mapping[str, object]], nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    constructor: ModelConstructor
    family: Literal["two_stage", "one_stage"]
    supported_weights: tuple[str, ...] = ("none", "imagenet1k_v1")
    backbone_weights: Mapping[str, object | None] = field(default_factory=dict)
