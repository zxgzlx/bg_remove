import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Type, Union
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from bg_remove import RMBG1
from bg_remove2 import RMBG2

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Background Removal Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web-assets")

MODEL_FACTORIES: Dict[int, Type[Union[RMBG1, RMBG2]]] = {
    1: RMBG1,
    2: RMBG2,
}
MODEL_CACHE: Dict[int, Optional[Union[RMBG1, RMBG2]]] = {1: None, 2: None}


def get_model(version: int):
    if version not in MODEL_FACTORIES:
        raise HTTPException(status_code=400, detail="Unsupported model version")
    if MODEL_CACHE[version] is None:
        MODEL_CACHE[version] = MODEL_FACTORIES[version]()
    return MODEL_CACHE[version]


def sanitize_filename(filename: Optional[str]) -> str:
    if filename:
        return Path(filename).name
    return f"result_{uuid4().hex}.png"


def bytes_to_data_url(data: bytes, content_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return bytes_to_data_url(buffer.getvalue(), "image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/remove-bg")
async def remove_background(
    model_version: int = Form(..., ge=1, le=2),
    image: UploadFile = File(...),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="缺少文件")

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="空文件无法处理")

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="无法识别的图片格式") from exc

    model = get_model(model_version)
    try:
        mask_img, result_img = model.remove_bg_image(pil_image)
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=f"抠图失败：{exc}") from exc

    original_content_type = image.content_type or "image/png"
    download_name = sanitize_filename(image.filename)

    return {
        "modelVersion": model_version,
        "originalDataUrl": bytes_to_data_url(contents, original_content_type),
        "maskDataUrl": image_to_data_url(mask_img),
        "resultDataUrl": image_to_data_url(result_img),
        "downloadName": download_name,
    }
