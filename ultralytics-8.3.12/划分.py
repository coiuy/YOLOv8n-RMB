import os
import random
from shutil import copy2

# 源文件夹路径
file_path = r"G:\标注后数据\UAVimg\imgs"
file_path_label = r"G:\标注后数据\UAVimg\labels"
# 新文件路径
new_file_path = r"G:\标注后数据\划分数据集后\images"
new_file_path_label = r"G:\标注后数据\划分数据集后\labels"

# 划分数据比例7:3
split_rate = [0.7, 0.3]

# 获取所有图片文件
all_images = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]

# 根据图片名匹配对应的标签
all_data = []
for img in all_images:
    label_name = os.path.splitext(img)[0] + '.txt'
    label_path = os.path.join(file_path_label, label_name)
    if os.path.exists(label_path):
        all_data.append((img, label_name))  # (图片, 标签)

# 打乱顺序
random.shuffle(all_data)

# 计算训练集和验证集的数量
train_count = int(len(all_data) * split_rate[0])
train_data = all_data[:train_count]
val_data = all_data[train_count:]

# 创建目标文件夹
for split_name in ['train', 'val']:
    os.makedirs(os.path.join(new_file_path, split_name), exist_ok=True)
    os.makedirs(os.path.join(new_file_path_label, split_name), exist_ok=True)

# 复制数据
def copy_dataset(data, img_target_path, label_target_path):
    for img_name, label_name in data:
        copy2(os.path.join(file_path, img_name), img_target_path)
        copy2(os.path.join(file_path_label, label_name), label_target_path)

copy_dataset(train_data, os.path.join(new_file_path, 'train'), os.path.join(new_file_path_label, 'train'))
copy_dataset(val_data, os.path.join(new_file_path, 'val'), os.path.join(new_file_path_label, 'val'))

print(f"Done! Train: {len(train_data)}, Val: {len(val_data)}")
