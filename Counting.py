from ultralytics.solutions import object_counter
import cv2

# 打开视频文件
cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\7月20日.mp4")
assert cap.isOpened(), "读取视频文件时出错"

# 读取第一帧以获取视频尺寸
ret, frame = cap.read()
if not ret:
    print("无法读取视频第一帧。请检查视频文件路径或内容。")
    exit(0)
height, width, _ = frame.shape

# 定义全屏区域点
region_points = [(0, 0), (0, height), (width, height), (width, 0)]

# --- 核心修改：在初始化 ObjectCounter 时传递所有参数 ---
# ObjectCounter 将会内部加载您指定的模型。
counter = object_counter.ObjectCounter(
    # 自定义模型路径作为 'model' 参数传递给 ObjectCounter
    model=r"G:\YOLOv8-Magic-8.3.12\YOLOv8-Magic-8.3.12\ultralytics-8.3.12\runs\train\exp23\weights\best.pt",
    # 传递计数区域点
    region=region_points,
    # 控制是否显示图像，这对应于 BaseSolution 中的 'show' 参数
    show=True,
    # 设置线条宽度
    line_width=2,
    # classes=None # 默认是 None，表示计数所有模型识别的类别
)

# 打印 ObjectCounter 内部加载的模型名称，这将是您 best.pt 的名称
print(f"计数器内部模型加载的类别名称 (counter.names): {counter.names}")
print(f"计数器设置的区域：{counter.region}")

counter.initialize_region()

# 开始视频处理循环
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频播放结束或无法读取更多帧。")
        break

    im0 = counter.count(im0)
    if counter.display_output(im0) is not None:
        break
cap.release()
cv2.destroyAllWindows()