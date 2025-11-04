Simple yolov8 comfyui plugin. This plugin adds Yolov8DetectionSegmentation node and ImageCompositeBlurredNode node.

Yolov8DetectionSegmentation detects and crops an input image.

ImageCompositeBlurredNode blurs image borders and overlaps the image at target location.

## Install

1. in comfyui `custom_nodes` dir and `https://github.com/dskjal/comfyui-yolov8-simple.git`
2. put detect or seg models in comfyui `models/yolov8` dir

## How to use

If your yolov8 model has "seg" or "Seg" or "SEG" in the name, the node outputs segmentation mask.

Node name is Yolov8DetectionSegmentation and ImageCompositeBlurredNode.


![](https://github.com/dskjal/comfyui-yolov8-simple/blob/main/simple-yolov8-workflow-example.png)


