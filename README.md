# [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)

## requirements

```bash
torch
torchvision
pillow
numpy
typing
scikit-image
hugginface_hub
transformers>=4.39.1
```

## Installation

```bash
pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt
```

## 通过 pipeline 加载

```python
from transformers import pipeline
image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
pillow_image = pipe(image_path) # applies mask on input and returns a pillow image
```

## 通过模型加载

```python
from PIL import Image
from skimage import io
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# prepare input
image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
orig_im = io.imread(image_path)
orig_im_size = orig_im.shape[0:2]
model_input_size = [1024, 1024]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference
result=model(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
pil_mask_im = Image.fromarray(result_image)
orig_image = Image.open(image_path)
no_bg_image = orig_image.copy()
no_bg_image.putalpha(pil_mask_im)
```

# [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

## requirements

```bash
torch
torchvision
pillow
kornia
transformers
```

## 用法

```python
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(input_image_path)
input_images = transform_image(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")

```
