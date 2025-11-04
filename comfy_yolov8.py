import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class Yolov8DSNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
                "class_id": ("INT", {"default": 0})
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "MASK", "IMAGE")
    RETURN_NAMES = ("cropped image", "x", "y", "width", "height", "mask", "debug image")
    FUNCTION = "detect"
    CATEGORY = "yolov8"

    def result_to_debug_image(self, results):
        im_array = results[0].plot()
        im = Image.fromarray(im_array[...,::-1])  # RGB PIL image

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)  # Convert back to CxHxW
        return torch.unsqueeze(image_tensor_out, 0)

    def detect(self, image, model_name, class_id):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image
        
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        results = model(image)

        image_np = np.asarray(image)
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_image_np = image_np[y1:y2, x1:x2]
                break   # TODO: only one image supported

        cropped_img_tensor_out = torch.tensor(cropped_image_np.astype(np.float32) / 255.0).unsqueeze(0)

        if "seg" in model_name or "Seg" in model_name or "SEG" in model_name:
            # segmentation mask
            masks = results[0].masks.data
            boxes = results[0].boxes.data

            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            class_indices = torch.where(clss == class_id)
            # use these indices to extract the relevant masks
            class_masks = masks[class_indices]
            mask_tensor = torch.any(class_masks, dim=0).int() * 255

        else:
            # box mask
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            mask_tensor = torch.tensor(mask).unsqueeze(0)  # (1, H, W)

        return (cropped_img_tensor_out, x1, y1, x2-x1, y2-y1, mask_tensor, self.result_to_debug_image(results))


NODE_CLASS_MAPPINGS = {
    "Yolov8DS": Yolov8DSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yolov8DS": "Yolov8DetectionSegmentation",
}