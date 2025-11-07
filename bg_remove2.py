from PIL import Image
import torch
from torchvision import transforms

from common import get_device, load_model

device = get_device()
model = load_model("models/briaai/RMBG-2.0", device)


def preprocess_image(image: Image, device: str):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_image(image).unsqueeze(0).to(device)


def model_inference(input_images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    return preds


def postprocess_image(result: torch.Tensor, im_size: list) -> Image:
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


# 输入图片路径
input_image_path = 'input/partner_1_0000.png'
# 加载图片
image = Image.open(input_image_path)
# 预处理图片
input_images = preprocess_image(image, device)
# 模型推理
result = model_inference(input_images)
# 后处理图片
mask = postprocess_image(result, image.size)
# 将图片和mask合并
image.putalpha(mask)
# 保存图片
image.save("output/no_bg_image_2.0.png")
