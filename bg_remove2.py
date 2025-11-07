import os
from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms

from common import get_device, get_file_name_without_ext, load_model


class RMBG2:
    model = None
    device = None

    def __init__(self):
        self.device = get_device()
        self.model = load_model("models/briaai/RMBG-2.0", self.device)

    def remove_bg(self, input_image_path: str, output_fold: str):
        image = Image.open(input_image_path)
        mask_img, result_img = self.remove_bg_image(image)

        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask_path = f"{output_fold}/{filename_without_exe}_mask_2.0.png"
        result_path = f"{output_fold}/{filename_without_exe}_2.0.png"
        # os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        # mask_img.save(mask_path)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        result_img.save(result_path)
        return {
            "original_path": input_image_path,
            "mask_path": mask_path,
            "result_path": result_path,
        }

    def batch_remove_bg(self, input_image, input_image_path: str, output_fold: str, input_suffix: str = ""):
        mask_img, result_img = self.remove_bg_image(input_image)

        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask_path = f"{output_fold}/rmgb2/mask/{filename_without_exe}/{filename_without_exe}{input_suffix}.png"
        result_path = f"{output_fold}/rmgb2/result/{filename_without_exe}/{filename_without_exe}{input_suffix}.png"
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

    def remove_bg_image(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        rgba_image = image.convert("RGBA")
        input_images = self.preprocess_image(
            rgba_image.convert("RGB"), self.device)
        result = self.model_inference(input_images)
        mask = self.postprocess_image(result, rgba_image.size)
        result_image = rgba_image.copy()
        result_image.putalpha(mask)
        return mask, result_image

    def preprocess_image(self, image: Image.Image, device: str):
        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
        return transform_image(image).unsqueeze(0).to(device)

    def model_inference(self, input_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        return preds

    def postprocess_image(self, result: torch.Tensor, im_size: Tuple[int, int]) -> Image.Image:
        r"""
        Convert model output to PIL image
        Args:
            result (torch.Tensor): Model output
        Returns:
            PIL.Image: PIL image
        """
        pred = result[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(im_size)
        return mask
