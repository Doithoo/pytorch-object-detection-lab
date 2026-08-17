# Model zoo

| Registry name | Role | Weight policies |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | Default lightweight learning detector | `none`, `imagenet1k_v1` |
| `fasterrcnn_resnet50_fpn` | Larger Faster R-CNN comparison | `none`, `imagenet1k_v1` |
| `ssdlite320_mobilenet_v3_large` | One-stage comparison | `none`, `imagenet1k_v1` |

`none` constructs without a download. ImageNet policies initialize only the supported backbone and preflight reports whether cache/network access is required. Checkpoint restoration always constructs with `none` before loading saved tensors.
