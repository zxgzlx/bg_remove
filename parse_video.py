import cv2
import sys
from pathlib import Path
from PIL import Image


class VideoToFrames:

    # ---------------------------------------------------------
    # ✅ 单个视频处理（含帧进度条）
    # ---------------------------------------------------------
    def handle_single_video(self, video_path: str, rmbg, output_folder: str):
        if rmbg is None:
            print("模型版本错误, 使用RMBG1.4模型")
            return

        video_name = Path(video_path).stem
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print(f"  错误: 无法打开视频文件 {video_path}")
            return

        # ---- ✅ 获取总帧数用于内部进度条 ----
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        frame_count = 0
        skip_frames = 0
        skip_frames_rate = 8

        print(f"  开始处理视频: {video_name} (总帧数: {total_frames})")

        # ========== ✅ 逐帧处理 ==========
        current_frame_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # 帧级计数（不跳）
            current_frame_index += 1

            # 跳帧逻辑
            is_skip = skip_frames % skip_frames_rate != 0
            skip_frames += 1
            if is_skip:
                # ✅ 即便跳帧也刷新进度条，让用户知道进度
                self._print_inner_progress(current_frame_index, total_frames)
                continue

            # ✅ 显示进度条
            self._print_inner_progress(current_frame_index, total_frames)

            # ---- ✅ 处理帧 ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)

            rmbg.batch_remove_bg(
                frame_image,
                f"{video_name}",
                output_folder,
                f"_{frame_count:04d}"
            )

            frame_count += 1

        video.release()
        print()  # 换行（结束内部条）

    # ---------------------------------------------------------
    # ✅ 批量视频处理（外部进度条）
    # ---------------------------------------------------------
    def handle_video_folder(self, video_folder_path: str, rmbg, output_folder: str):
        if rmbg is None:
            print("Invalid model version")
            return

        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
        video_folder_path = Path(video_folder_path)

        video_paths = [
            p for p in video_folder_path.iterdir()
            if p.suffix.lower() in video_extensions
        ]

        total = len(video_paths)
        if total == 0:
            print("未找到可处理的视频文件")
            return

        print(f"找到 {total} 个视频，开始处理...\n")

        # ====================================
        # ✅ 外层视频级进度条
        # ====================================
        for idx, video_path in enumerate(video_paths, start=1):

            self._print_outer_progress(idx, total)

            print(f"\n处理视频: {video_path}")
            self.handle_single_video(video_path, rmbg, output_folder)

        print("\n✅ 所有视频处理完成 ✅")

    # ---------------------------------------------------------
    # ✅ 工具方法：外层进度条
    # ---------------------------------------------------------
    def _print_outer_progress(self, idx, total):
        bar_len = 40
        progress = idx / total
        filled = int(bar_len * progress)
        bar = "█" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r视频进度：[{bar}] {idx}/{total} ({progress*100:.1f}%)")
        sys.stdout.flush()

    # ---------------------------------------------------------
    # ✅ 工具方法：内层帧进度条
    # ---------------------------------------------------------
    def _print_inner_progress(self, current, total):
        bar_len = 40
        progress = current / total
        filled = int(bar_len * progress)
        bar = "█" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r    帧进度：[{bar}] {current}/{total} ({progress*100:.1f}%)")
        sys.stdout.flush()
