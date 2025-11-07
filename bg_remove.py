from PIL import Image
import numpy as np
from skimage import io
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

    def remove_bg(self, input_image_path):
        # 第一次读取图片，获取图片尺寸
        orig_im = io.imread(input_image_path)
        orig_im_size = orig_im.shape[0:2]
        # 预处理图片
        image = self.preprocess_image(orig_im, self.device)

        # 模型推理
        result = self.model_inference(image)

        # 后处理获取遮罩数据
        mask_ndarray = self.postprocess_image(result[0][0], orig_im_size)

        # 创建mask图片
        mask = Image.fromarray(mask_ndarray)
        filename_without_exe = get_file_name_without_ext(input_image_path)
        mask.save(f"output/{filename_without_exe}_mask_1.4.png")
        # 将图片和mask合并
        orig_image = Image.open(input_image_path)
        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(mask)
        # 保存图片
        no_bg_image.save(f"output/{filename_without_exe}_1.4.png")

    def preprocess_image(self, im: np.ndarray, device: str) -> torch.Tensor:
        image_size = [1024, 1024]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(
            im_tensor, 0), size=image_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image.to(device)

    def model_inference(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = torch.squeeze(F.interpolate(
            result, size=im_size, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_array = (result*255).permute(1, 2,
                                        0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array
