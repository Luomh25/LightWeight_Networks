import os
from PIL import Image

# 原始大文件夹路径和目标大文件夹路径
original_folder = '/root/0424/Poly/dataset'
target_folder = '/root/0424/Poly/vessel'

# 遍历原始大文件夹中的所有子文件夹
for root, dirs, files in os.walk(original_folder):
    for dir_name in dirs:
        original_subfolder = os.path.join(root, dir_name)
        target_subfolder = os.path.join(target_folder, os.path.relpath(original_subfolder, original_folder))

        # 创建对应的目标子文件夹
        os.makedirs(target_subfolder, exist_ok=True)

        # 遍历子文件夹中的所有图像文件
        for file_name in os.listdir(original_subfolder):
            file_path = os.path.join(original_subfolder, file_name)

            # 打开图像文件
            image = Image.open(file_path)

            # 将图像从灰度格式转换为 RGB 格式
            rgb_image = image.convert("RGB")

            # 保存转换后的图像到目标子文件夹中
            target_file_path = os.path.join(target_subfolder, file_name)
            rgb_image.save(target_file_path)

            print("Converted and saved:", target_file_path)