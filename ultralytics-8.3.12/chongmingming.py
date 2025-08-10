import os

def rename_images(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # 获取所有图片文件
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(image_extensions)]

    files.sort()

    # 第一步：先将所有文件重命名为临时唯一名，避免冲突
    temp_names = []
    for idx, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        tmp_name = f"__tmp_{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, tmp_name)
        os.rename(src, dst)
        temp_names.append(tmp_name)

    # 第二步：统一按img序号命名
    for idx, tmp_name in enumerate(temp_names, 1):
        ext = os.path.splitext(tmp_name)[1]
        new_name = f"img{idx}{ext}"
        src = os.path.join(folder_path, tmp_name)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {tmp_name} -> {new_name}")

if __name__ == "__main__":
    target_folder = r"F:\数据集\DataSet\data from Dehong city"
    rename_images(target_folder)
