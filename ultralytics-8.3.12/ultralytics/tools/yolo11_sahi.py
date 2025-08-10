# 导入必要的库
import os
import shutil

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from IPython.display import Image, display

# 创建目录用于存放结果
os.makedirs('sahi_detect_result', exist_ok=True)

# ===== 您可以在此处输入您的模型和本地图像路径 =====
model_path = "ultralytics-8.3.12/yolo11n.pt"     # 替换为您的模型文件路径
image_path = "ultralytics-8.3.12/ultralytics/assets/bus.jpg"    # 替换为您的本地图像文件路径

# 检查模型和图像文件是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到：{model_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图像文件未找到：{image_path}")

# 实例化检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",       # 根据您的模型类型进行修改，例如 'yolov5'
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0",              # 如果有 GPU，可改为 'cuda:0'
)

# 标准推理
result = get_prediction(image_path, detection_model)

# 导出并显示标准推理结果
export_dir_standard = os.path.join("sahi_detect_result", "standard_prediction")
result.export_visuals(export_dir=export_dir_standard)
standard_prediction_image_path = os.path.join(export_dir_standard, os.listdir(export_dir_standard)[0])
shutil.move(standard_prediction_image_path, os.path.join("sahi_detect_result", "standard_prediction.png"))
display(Image("sahi_detect_result/standard_prediction.png"))

# 切片推理
result_sliced = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 导出并显示切片推理结果
export_dir_sliced = os.path.join("sahi_detect_result", "sliced_prediction")
result_sliced.export_visuals(export_dir=export_dir_sliced)
sliced_prediction_image_path = os.path.join(export_dir_sliced, os.listdir(export_dir_sliced)[0])
shutil.move(sliced_prediction_image_path, os.path.join("sahi_detect_result", "sliced_prediction.png"))
display(Image("sahi_detect_result/sliced_prediction.png"))
