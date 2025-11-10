import subprocess
from pathlib import Path


def pngs_to_webm(input_dir: str, output_path: str, fps: int = 30, with_alpha: bool = True):
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} 不是有效目录")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"{input_dir} 下没有找到任何 .png 文件")

    # 写入绝对路径到 list.txt
    list_file = input_dir / "png_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in png_files:
            f.write(f"file '{p.as_posix()}'\n")

    # 输出路径也用绝对路径
    output_path = str(Path(output_path).resolve())

    # 构建命令（全路径）
    cmd = [
        "ffmpeg",
        "-y",
        "-r", str(fps),
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file.as_posix()),
    ]

    if with_alpha:
        cmd += ["-c:v", "libvpx-vp9", "-pix_fmt",
                "yuva420p", "-auto-alt-ref", "0"]
    else:
        cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p"]

    cmd += [output_path]

    print("运行命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ 完成！输出文件：", output_path)


if __name__ == "__main__":
    # 示例：将 ./xxx 目录下的 PNG 合成 out.webm
    pngs_to_webm(
        input_dir="input/partner_0",         # 你的 png 目录
        output_path="output/output.webm",
        fps=30,
        with_alpha=True,        # 如果不需要透明，改为 False
    )
