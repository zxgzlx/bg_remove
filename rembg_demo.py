from rembg import remove
from PIL import Image
from rembg.session_factory import new_session

# 创建 isnet-general-use 会话
session = new_session("isnet-general-use")

# 1️⃣ 打开输入图片
input_path = "input/partner_0_0000.png"
output_path = "output/partner_0_0000.png"

# 2️⃣ 打开并统一为 RGB
img = Image.open(input_path).convert("RGB")

# 3️⃣ 调用 ISNet 通用模型
out = remove(img, session=session)

# 4️⃣ 保存结果（透明背景 PNG）
out.save(output_path)

print(f"✅ Saved to {output_path}")
