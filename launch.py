import json
from pathlib import Path

from bg_remove import RMBG1
from bg_remove2 import RMBG2
from parse_video import VideoToFrames

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"配置文件解析失败，使用默认配置: {exc}")
        return {}


def main():
    config = load_config()

    image_path = config.get("input_image_path")
    # image_path = "input/partner_1_0000.png"
    model_index = config.get("model_index")
    print("model_index = ", model_index)
    type = config.get("type")
    print("type = ", type)
    output_folder = config.get("output_folder")

    rmbg = None
    if model_index == 1:
        # 使用RMBG1.4模型
        print("Using RMBG1...")
        rmbg = RMBG1()
    elif model_index == 2:
        # 使用RMBG2.0模型
        print("Using RMBG2...")
        rmbg = RMBG2()

    if rmbg is None:
        print(f"Invalid model index {model_index}")
        return
    if type == "image":
        if not image_path:
            print("缺少图片路径，请通过命令行或 config.json 的 input_image_path 指定")
            return
        rmbg.remove_bg(image_path, output_folder)
    elif type == "video":
        video_handler = VideoToFrames()
        input_video_path = config.get("input_video_path")
        print(f"input_video_path: {input_video_path}")
        # 判断是目录模式还是文件模式
        if Path(input_video_path).is_dir():
            video_handler.handle_video_folder(
                input_video_path, rmbg, output_folder)
        else:
            video_handler.handle_single_video(
                input_video_path, rmbg, output_folder)


if __name__ == '__main__':
    # VideoToFrames().handle_video_folder("input/videos", RMBG2())
    main()
