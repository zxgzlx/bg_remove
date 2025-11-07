import os
import torch
from transformers import AutoModelForImageSegmentation


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
