from PIL import Image
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from common import get_device, load_model

device = get_device()
model = load_model("models/briaai/RMBG-1.4", device)


def preprocess_image(im: np.ndarray, device: str) -> torch.Tensor:
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


def model_inference(image: torch.Tensor) -> torch.Tensor:
    return model(image)


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(
        result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1, 2,
                                    0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


# prepare input
image_path = "input/partner_1_0000.png"
orig_im = io.imread(image_path)
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, device)

# inference
result = model_inference(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
pil_mask_im = Image.fromarray(result_image)
orig_image = Image.open(image_path)
no_bg_image = orig_image.copy()
no_bg_image.putalpha(pil_mask_im)
no_bg_image.save("output/no_bg_image_1.4.png")
