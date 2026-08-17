# Metrics

`map_50_95` averages AP over IoU 0.50:0.05:0.95; `map_50` and `map_75` are AP at fixed IoU thresholds. `mar_1`, `mar_10`, and `mar_100` cap detections per image. Per-class output reports `map_50_95` and `mar_100`. Backend negative sentinels become numeric zero; in-memory values are not rounded, while JSON/CSV uses six decimals.

AP consumes backend predictions without a display threshold. Error analysis separately filters by `error_score_threshold` and greedily matches same-class ordinary targets at `error_iou_threshold`; difficult-only matches are ignored. Use validation metrics for choices. Keep VOC test reserved for one final report. Bounded or synthetic scores are not full-VOC benchmarks.
