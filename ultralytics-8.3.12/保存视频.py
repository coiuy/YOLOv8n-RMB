from ultralytics.solutions import object_counter
import cv2
import os

# 输入视频路径和输出视频路径
input_video = r"F:\数据集\UAV数据集\8月9日 (1).mp4"
output_video = r"F:\数据集\UAV数据集\CountingVideo1.mp4"

# 打开视频文件
cap = cv2.VideoCapture(input_video)
assert cap.isOpened(), "读取视频文件时出错"

# 读取第一帧以获取视频尺寸和帧率
ret, frame = cap.read()
if not ret:
    print("无法读取视频第一帧。请检查视频文件路径或内容。")
    exit(0)
height, width, _ = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(os.path.dirname(output_video), exist_ok=True)
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 定义全屏区域点
region_points = [(0, 0), (0, height), (width, height), (width, 0)]

# 初始化 ObjectCounter
counter = object_counter.ObjectCounter(
    model=r"G:\YOLOv8-Magic-8.3.12\YOLOv8-Magic-8.3.12\ultralytics-8.3.12\runs\train\exp23\weights\best.pt",
    region=region_points,
    show=False,  # 禁用弹出窗口
    line_width=2,
)

print(f"计数器内部模型加载的类别名称 (counter.names): {counter.names}")
print(f"计数器设置的区域：{counter.region}")

counter.initialize_region()

# 重置视频到第一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 开始视频处理循环
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频播放结束或无法读取更多帧。")
        break

    im0 = counter.count(im0)  # 处理图像
    video_writer.write(im0)   # 将结果写入视频

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"处理完成！结果视频已保存到：{output_video}")
