import argparse
from bg_remove import RMBG1
from bg_remove2 import RMBG2


def main():
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="Background remover")
    # parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--version", "-v", type=int, default=1,
                        choices=[1, 2],
                        help="Select model version: 1 for RMBG1, 2 for RMBG2")

    args = parser.parse_args()

    # image_path = args.image_path
    image_path = "input/partner_1_0000.png"
    version = args.version

    if version == 1:
        # 使用RMBG1.4模型
        print("Using RMBG1...")
        RMBG1(image_path)
    elif version == 2:
        # 使用RMBG2.0模型
        print("Using RMBG2...")
        RMBG2(image_path)


if __name__ == '__main__':

    main()
