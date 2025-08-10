from ultralytics import YOLO

if __name__ == "__main__":
    # 加载模型
    # model = YOLO(r"G:\YOLOv8-Magic-8.3.12\YOLOv8-Magic-8.3.12\ultralytics-8.3.12\runs\train\YOLOv8n-RMB\weights\best.pt")  # YOLOv8n模型
    model = YOLO(r"G:\YOLOv8-Magic-8.3.12\YOLOv8-Magic-8.3.12\ultralytics-8.3.12\runs\train\exp23\weights\best.pt")
    model.predict(
        source=r"F:\数据集\UAV数据集\UAVimg\imgs\img175.jpg",    #F:\数据集\德宏数据集\橡胶数据集1\橡胶数据集1
        save=True,  # 保存预测结果
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project="runs/predict",  # 项目名称（可选）
        name="exp",  # 实验名称，结果保存在'project/name'目录下（可选）09
        save_txt=False,  # 保存结果为 .txt 文件
        save_conf=True,  # 保存结果和置信度分数
        save_crop=False,  # 保存裁剪后的图像和结果
        show_labels=True,  # 在图中显示目标标签
        show_conf=True,  # 在图中显示目标置信度分数
        vid_stride=20,  # 视频帧率步长
        line_width=5,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=False,  # 使用高分辨率的分割掩码
        show_boxes=True,  # 在分割预测中显示边界框
    )
