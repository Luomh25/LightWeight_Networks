import os
import shutil

# 设置源文件夹和目标文件夹路径
source_folder = 'D:/Uni/Project/0424/Poly/vessels'  # 输入源文件夹路径
target_folder = 'D:/Uni/Project/0426/Poly'  # 输入目标文件夹路径

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 定义每个子文件夹中的图像数量
images_per_folder = 6

# 遍历图像文件
for i, filename in enumerate(sorted(os.listdir(source_folder))):
    if filename.endswith('.jpg'):
        source_file = os.path.join(source_folder, filename)

        # 计算目标子文件夹编号
        folder_index = (i // images_per_folder) + 1

        # 创建目标子文件夹
        folder_path = os.path.join(target_folder, str(folder_index))
        os.makedirs(folder_path, exist_ok=True)

        # 将图像文件移动到目标子文件夹
        target_file = os.path.join(folder_path, filename)
        shutil.copy(source_file, target_file)

        print(f'Moved {filename} to folder {folder_index}')
'''
import os
import shutil

# 设置源文件夹和目标文件夹路径
source_folder = 'D:/Uni/Project/0424/vein_ml'  # 输入源文件夹路径
target_folder = 'D:/Uni/Project/0415/Tongji_enhance_2/enhanced'  # 输入目标文件夹路径

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历图像文件
for folder_name in sorted(os.listdir(source_folder)):
    folder_path = os.path.join(source_folder, folder_name)
    if os.path.isdir(folder_path):
        # 遍历目标子文件夹中的图像文件
        for filename in sorted(os.listdir(folder_path)):
            source_file = os.path.join(folder_path, filename)

            # 将图像文件复制回源文件夹
            target_file = os.path.join(target_folder, filename)
            shutil.copyfile(source_file, target_file)

            print(f'Copied {filename} back to target folder')
'''
