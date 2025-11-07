from PIL import Image
import torch
from torchvision import transforms

from common import get_device, get_file_name_without_ext, load_model


class RMBG2:
    model = None

    def __init__(self, input_image_path):
        device = get_device()
        self.model = load_model("models/briaai/RMBG-2.0", device)
        # 加载图片
        image = Image.open(input_image_path)
        # 预处理图片
        input_images = self.preprocess_image(image, device)
        # 模型推理
        result = self.model_inference(input_images)
        # 后处理图片
        mask = self.postprocess_image(result, image.size)
        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask.save(f"output/{filename_without_exe}_mask_2.0.png")
        # 将图片和mask合并
        image.putalpha(mask)
        # 保存图片
        image.save(f"output/{filename_without_exe}_2.0.png")

    def preprocess_image(self, image: Image, device: str):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_image(image).unsqueeze(0).to(device)

    def model_inference(self, input_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        return preds

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> Image:
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
