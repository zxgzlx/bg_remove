import base64
import json
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from uuid import uuid4

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from bg_remove import RMBG1
from bg_remove2 import RMBG2
import launch

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
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
CONFIG_DEFAULTS: Dict[str, Any] = {
    "model_index": 1,
    "type": "image",
    "input_image_path": "",
    "input_video_path": "",
    "output_folder": "output",
}
CONFIG_KEYS = set(CONFIG_DEFAULTS.keys())


def get_model(version: int):
    if version not in MODEL_FACTORIES:
        raise HTTPException(
            status_code=400, detail="Unsupported model version")
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


def load_config() -> Dict[str, Any]:
    data = CONFIG_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if key in CONFIG_KEYS:
                        data[key] = value
        except json.JSONDecodeError:
            pass
    return data


def save_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = load_config()
    for key, value in payload.items():
        if key in CONFIG_KEYS:
            config[key] = value
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    return config


def pick_path(dialog_type: str) -> str:
    try:
        from tkinter import filedialog, Tk  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("当前环境不支持 tkinter") from exc

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if dialog_type == "directory":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename()
    finally:
        root.destroy()
    return path or ""


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
def update_config(config: Dict[str, Any] = Body(...)):
    return save_config(config or {})


@app.post("/api/pick-path")
async def pick_config_path(payload: Dict[str, Any] = Body(...)):
    dialog_type = payload.get("mode", "file")
    if dialog_type not in {"file", "directory"}:
        raise HTTPException(
            status_code=400, detail="mode 参数必须为 file 或 directory")

    try:
        path = await run_in_threadpool(partial(pick_path, dialog_type))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"无法打开资源管理器: {exc}") from exc

    return {"path": path}


@app.post("/api/run-launch")
async def run_launch(config_override: Optional[Dict[str, Any]] = Body(default=None)):
    to_merge = config_override or {}
    if to_merge:
        save_config(to_merge)
    try:
        await run_in_threadpool(launch.main)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"执行失败：{exc}") from exc
    return {"status": "ok", "message": "处理完成"}


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
