from PIL import Image
import os

# 要转换的文件名
input_files = ['ir.png', 'vis.png']

# 遍历文件进行转换
for file_name in input_files:
    # 打开图像文件
    with Image.open(file_name) as img:
        # 获取文件名和扩展名
        base_name, _ = os.path.splitext(file_name)
        # 转换并保存为JPEG格式
        img.convert('RGB').save(f"{base_name}.jpg", 'JPEG')

print("图像转换完成")