import os
import torch
from transformers import AutoModelForImageSegmentation
from PIL import Image
import cv2
import numpy as np


def get_device() -> str:
    r"""
    获取cuda或cpu, 如果存在GPU则返回GPU，不存在则返回CPU

    Returns:
        str: cuda or cpu
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 判断device是否为GPU
    if device == "cuda":
        print("GPU is running")
    else:
        print("CPU is running")
    return device


def load_model(pretrained_model_name_or_path: str, device: str) -> AutoModelForImageSegmentation:
    r"""
    加载模型

    Args:
        pretrained_model_name_or_path (str): 模型名称或路径
        device (str): 设备名称

    Returns:
        AutoModelForImageSegmentation: 模型
    """
    return AutoModelForImageSegmentation.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, local_files_only=True).eval().to(device)


def get_file_name_without_ext(file_path: str) -> str:
    r"""
    获取文件名（不带扩展名）

    Args:
        file_path (str): 文件路径

    Returns:
        str: 文件名（不带扩展名）
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_file_name_with_ext(file_path: str) -> str:
    r"""
    获取文件名（带扩展名）

    Args:
        file_path (str): 文件路径

    Returns:
        str: 文件名（带扩展名）
    """
    return os.path.basename(file_path)


def full_decontaminate(img: Image.Image, bg_rgb=(1, 1, 1),
                       strength=1.0, gamma=0.9, shrink=0.2) -> Image.Image:
    """
    针对 RMBG 白边的完全去污染算法。
    1. 颜色反混合 (un-premultiply)
    2. 边缘收缩 + gamma 调整
    3. 输出干净 RGBA
    """
    arr = np.array(img).astype(np.float32) / 255.0
    rgb, a = arr[..., :3], arr[..., 3:4]
    safe_a = np.maximum(a, 1e-6)

    bg = np.array(bg_rgb).reshape(1, 1, 3)
    # 反混合：去掉白色背景影响
    fg = (rgb - bg * (1 - safe_a)) / safe_a
    fg = np.clip(fg, 0, 1)

    # 在原 rgb 与修正后之间插值
    rgb = rgb * (1 - strength) + fg * strength
    rgb = np.clip(rgb, 0, 1)

    # 收紧 α
    if shrink > 0:
        a = np.maximum(0, (a - shrink) / (1 - shrink))
    if gamma != 1.0:
        a = np.power(a, gamma)

    # 去透明区残留颜色
    mask = (a.squeeze(-1) < 0.05)  # (H, W)
    rgb[mask] = 0

    out = np.concatenate([rgb, a], axis=-1)
    return Image.fromarray((out * 255).astype(np.uint8), "RGBA")


def clean_rmbg_halo(img: Image.Image,
                    bg_rgb=(1, 1, 1),
                    strength=0.9,
                    alpha_gamma=0.9,
                    alpha_shrink=0.03,
                    clip_low_alpha=True,
                    premultiply=True) -> Image.Image:
    """
    多步骤 RMBG 输出修复：
    1. 去除背景颜色污染；
    2. 透明区域清理；
    3. 调整 alpha（收紧边缘）；
    4. 转预乘 alpha，避免叠加发白。
    """
    arr = np.array(img).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    safe_a = np.maximum(a, 1e-6)

    # Step 1. 去除背景残色
    bg = np.array(bg_rgb).reshape(1, 1, 3)
    corrected = (rgb - bg * (1 - safe_a)) / safe_a
    rgb = rgb*(1-strength) + corrected*strength
    rgb = np.clip(rgb, 0, 1)

    # Step 2. 清理透明区 RGB（α≈0时的背景残留）
    if clip_low_alpha:
        mask = (a.squeeze(-1) < 0.05)  # (H, W)
        rgb[mask] = 0

    # Step 3. 调整 alpha 边缘（gamma + shrink）
    if alpha_shrink > 0:
        a = np.maximum(0, (a - alpha_shrink) / (1 - alpha_shrink))
    if alpha_gamma != 1.0:
        a = np.power(a, alpha_gamma)
    a = np.clip(a, 0, 1)

    # Step 4. 转预乘 alpha（premultiply）
    if premultiply:
        rgb *= a

    out = np.concatenate([rgb, a], axis=-1)
    return Image.fromarray((out * 255).astype(np.uint8), "RGBA")


def refine_alpha_edges(
    rgba_img: Image.Image, erode_px=1, blur_px=1, dilate_px=0
) -> Image.Image:
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]
    if erode_px > 0:
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
    if blur_px > 0:
        alpha = cv2.GaussianBlur(alpha, (blur_px * 2 + 1, blur_px * 2 + 1), 0)
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def decontaminate_halo(
    rgba_img: Image.Image, bg_rgb01=(1, 1, 1), strength=0.8
) -> Image.Image:
    arr = np.array(rgba_img).astype(np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4]
    bg = np.array(bg_rgb01).reshape(1, 1, 3)
    corrected = (rgb - bg * (1 - alpha)) / np.maximum(alpha, 1e-6)
    mixed = rgb * (1 - strength) + corrected * strength
    mixed = np.clip(mixed, 0, 1)
    result = np.concatenate([mixed, alpha], axis=2)
    return Image.fromarray((result * 255).astype(np.uint8), "RGBA")
