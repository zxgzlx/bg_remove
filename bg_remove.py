import os
from typing import Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from common import get_device, get_file_name_without_ext, load_model


class RMBG1:
    model = None
    device = None

    def __init__(self):
        self.device = get_device()
        self.model = load_model("models/briaai/RMBG-1.4", self.device)

    def remove_bg(self, input_image_path: str):
        orig_image = Image.open(input_image_path)
        mask_img, result_img = self.remove_bg_image(orig_image)

        os.makedirs("output", exist_ok=True)
        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask_path = f"output/{filename_without_exe}_mask_1.4.png"
        result_path = f"output/{filename_without_exe}_1.4.png"
        # mask_img.save(mask_path)
        result_img.save(result_path)
        return {
            "original_path": input_image_path,
            "mask_path": mask_path,
            "result_path": result_path,
        }

    def batch_remove_bg(self, input_image, input_image_path: str, suffix: str = ""):
        mask_img, result_img = self.remove_bg_image(input_image)

        os.makedirs("output", exist_ok=True)
        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask_path = f"output/rmgb1/mask/{filename_without_exe}/{filename_without_exe}{suffix}.png"
        result_path = f"output/rmgb1/result/{filename_without_exe}/{filename_without_exe}{suffix}.png"
        # 如果result_path的父目录不存在，则创建
        # os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        # mask_img.save(mask_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        result_img.save(result_path)
        return {
            "original_path": input_image_path,
            "mask_path": mask_path,
            "result_path": result_path,
        }

    def remove_bg_image(self, orig_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        rgb_image = orig_image.convert("RGB")
        rgba_image = orig_image.convert("RGBA")
        orig_im = np.array(rgb_image)
        orig_im_size = orig_im.shape[0:2]
        image = self.preprocess_image(orig_im, self.device)

        result = self.model_inference(image)
        mask_ndarray = self.postprocess_image(result[0][0], orig_im_size)
        mask = Image.fromarray(mask_ndarray)
        result_image = rgba_image.copy()
        result_image.putalpha(mask)
        return mask, result_image

    def preprocess_image(self, im: np.ndarray, device: str) -> torch.Tensor:
        image_size = [1024, 1024]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0), size=image_size, mode="bilinear"
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image.to(device)

    def model_inference(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = torch.squeeze(F.interpolate(
            result, size=im_size, mode="bilinear"), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (
            (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        )
        im_array = np.squeeze(im_array)
        return im_array
