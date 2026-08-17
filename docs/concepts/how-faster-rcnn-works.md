# How Faster R-CNN works

The backbone extracts features and FPN exposes several resolutions. The RPN scores anchors and proposes regions. ROI Align samples each region; ROI heads classify foreground/background and regress box corrections. Training jointly optimizes proposal and ROI losses. Inference applies scores and suppression inside torchvision before returning final detections.
