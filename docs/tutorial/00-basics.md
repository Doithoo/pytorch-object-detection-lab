# Tutorial 00: Images, Boxes, Labels, and Batches

[Simplified Chinese](00-basics.zh-CN.md) | [Tutorial index](README.md)

This chapter introduces the tensor format used by every later data, model,
training, and evaluation step. You need basic Python indexing and tensor shapes;
you do not need VOC data, a checkpoint, or network access.

## One image, one target dictionary

A detection sample is a pair, not a single tensor:

```text
image                       float32 [3, H, W], values in [0, 1]
target["boxes"]             float32 [N, 4]
target["labels"]            int64   [N]
target["image_id"]          int64   [1]
target["area"]              float32 [N]
target["iscrowd"]           int64   [N]
target["difficult"]         bool    [N]
```

The leading `N` must agree across all object fields. `N` is allowed to be zero.
An empty target still has `boxes` shaped `[0, 4]`, not `[0]`, and the remaining
object fields are shaped `[0]`. This matters for images with no ordinary objects
and for training samples whose only VOC annotations are marked difficult.

Labels are class indices, not arbitrary IDs. Torchvision reserves label `0` for
background. This project maps the 20 VOC object classes to `1..20`; target boxes
therefore carry foreground labels only. The detector learns background from
proposals that do not match a foreground target. Do not add a made-up background
box with label `0`.

## Compute an xyxy box by hand

The project uses zero-based continuous `xyxy` coordinates:

```text
[x1, y1, x2, y2] = [left, top, right, bottom]
width  = x2 - x1
height = y2 - y1
area   = width * height
```

For the first box in the example, `[2, 3, 18, 15]`, the width is `18 - 2 =
16`, the height is `15 - 3 = 12`, and the area is `192`. The second box
`[20, 4, 30, 22]` has area `(30 - 20) * (22 - 4) = 180`.

Run the calculation:

```bash
uv run python examples/01_boxes_and_labels.py
```

Expected output contains two boxes, `labels=[1, 3]`, and
`areas=[192.0, 180.0]`. Change one coordinate, predict the new area on paper,
and run the example again. A box is invalid when `x2 <= x1` or `y2 <= y1`.

## Why a detection batch is two lists

Classification tutorials often stack images into `[B, 3, H, W]`. Detection
images keep their native sizes before entering the model, so an image
`[3, 16, 20]` cannot be stacked directly with `[3, 12, 24]`. Object counts also
differ. The collate function therefore returns:

```text
images  = list[Tensor[3, Hi, Wi]]
targets = list[dict[str, Tensor]]
len(images) == len(targets) == batch size
```

Run the real project collate function:

```bash
uv run python examples/02_detection_batch.py
```

Expected output is:

```text
image_shapes=[(3, 16, 20), (3, 12, 24)]
target_counts=[1, 2]
```

Torchvision performs its own resize, normalization, and padding after receiving
this list. Do not pre-stack unequal images or invent padding without also
tracking the original image sizes needed to return boxes to image coordinates.

## IoU is overlap, not score

Intersection over Union (IoU) compares two boxes geometrically. For boxes
`A=[10,10,30,30]` and `B=[20,10,40,30]`, each area is `400`. Their intersection
is `10 * 20 = 200`, so the union is `400 + 400 - 200 = 600` and IoU is `1/3`.
A model confidence score answers a different question and cannot replace IoU.
Chapter 05 uses both values with separate thresholds.

## Common mistakes

- `boxes` has integer dtype, the wrong final dimension, or non-positive width or
  height: data validation/model code fails before meaningful learning.
- `labels` uses float values or includes background `0`: the class values are
  wrong even if tensor shapes look plausible.
- Images are stacked before collation: variable-size samples cannot form one
  rectangular tensor.
- An empty annotation becomes `torch.tensor([])`: its shape is `[0]`, so reshape
  boxes explicitly to `[0, 4]`.
- A printed area matches only because coordinates were treated as inclusive:
  this project uses continuous boundaries and no `+1` term.

Next, make the Python and device environment reproducible in
[Tutorial 01](01-environment.md). If the environment already works, continue to
[VOC data and boxes](02-data-and-boxes.md).
