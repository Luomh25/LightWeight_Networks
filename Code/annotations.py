import os
import json

root_dir = 'D:/Uni/Project/0415/Tongji_enhance_2/datasets'  # 数据集根目录路径
image_size = (128, 128)  # 图像尺寸
num_images_per_hand = 10  # 每个手掌的图像数量
num_hands = 600  # 手掌的数量

annotations = []

for hand_id in range(1, num_hands + 1):
    start_index = (hand_id - 1) * num_images_per_hand + 1
    end_index = hand_id * num_images_per_hand

    hand_annotation = {
        'hand_id': hand_id,
        'images': []
    }

    for image_index in range(start_index, end_index + 1):
        image_id = str(image_index).zfill(5)
        image_path = os.path.join(root_dir, image_id + '.bmp')
        label = hand_id

        image_info = {
            'file_name': image_id + '.bmp',
            'width': image_size[0],
            'height': image_size[1],
            'label': label
        }

        hand_annotation['images'].append(image_info)

    annotations.append(hand_annotation)

# 保存注释为JSON文件
annotation_file = os.path.join(root_dir, 'annotations.json')
with open(annotation_file, 'w') as f:
    json.dump(annotations, f)

print("Annotations saved to:", annotation_file)