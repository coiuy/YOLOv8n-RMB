#!/usr/bin/env python3
# coding=utf-8

from ultralytics import YOLO

def train_multiple_models():
    """
    此函数会遍历模型列表，依次使用你指定的 model YAML 和实验名称进行训练。
    你只需修改 model_configs 中的 yaml_path 和 exp_name 即可。
    其他训练超参数会保持一致。
    """

    # 1. 在这里定义你所有要训练的模型配置（model 路径 + 实验名称）
    model_configs = [
        {
            "yaml_path": "/home/q/qitjwang/Github/YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/v8v3spd.yaml",
            "exp_name": "exp_v3spd"
        },
        {
            "yaml_path": "/home/q/qitjwang/Github/YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/models/v8/v8v3tiny.yaml",
            "exp_name": "exp_v3tiny"
        },
        # 你可以继续在这里增加更多的模型配置
        # {
        #     "yaml_path": "/path/to/another_model.yaml",
        #     "exp_name": "exp_another"
        # },
    ]

    # 2. 训练参数基本保持一致，这里仅示例放了少量关键参数，你可根据需要改动
    #    如果要改动，就直接在下方的 model.train() 里调整。
    for config in model_configs:
        print(f"开始训练模型：{config['yaml_path']} ；实验名：{config['exp_name']}")

        # 加载对应的模型配置（不使用预训练权重）
        model = YOLO(config["yaml_path"])
        
        # 开始训练
        model.train(
            data="/home/q/qitjwang/Github/YOLOv8-Magic/ultralytics-8.3.12/ultralytics/cfg/datasets/coco128.yaml",
            epochs=100,  # (int) 训练的周期数
            patience=0,  # (int) 等待无明显改善以进行早期停止的周期数
            batch=32,  # (int) 每批次的图像数量（-1 为自动批处理）
            imgsz=640,  # (int) 输入图像的大小，整数或w，h
            val=True,  # (bool) 在训练期间进行验证/测试
            save=True,  # (bool) 保存训练检查点和预测结果
            save_period=40,  # (int) 每x周期保存检查点（如果小于1则禁用）
            cache=True,  # (bool) True/ram、磁盘或False。使用缓存加载数据
            device=0,  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
            workers=8,  # (int) 数据加载的工作线程数（每个DDP进程）
            project="runs/train",  # (str, optional) 项目名称
            name="exp",  # (str, optional) 实验名称，结果保存在'project/name'目录下
            exist_ok=False,  # (bool) 是否覆盖现有实验
            pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
            optimizer="SGD",  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
            verbose=True,  # (bool) 是否打印详细输出
            seed=3,  # (int) 用于可重复性的随机种子
            deterministic=True,  # (bool) 是否启用确定性模式
            single_cls=False,  # (bool) 将多类数据训练为单类
            rect=False,  # (bool) 如果mode='train'，则进行矩形训练，如果mode='val'，则进行矩形验证
            cos_lr=False,  # (bool) 使用余弦学习率调度器
            close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
            resume=False,  # (bool) 从上一个检查点恢复训练
            amp=False,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
            fraction=1.0,  # (float) 要训练的数据集分数（默认为1.0，训练集中的所有图像）
            profile=False,  # (bool) 在训练期间为记录器启用ONNX和TensorRT速度
            freeze=None,  # (int | list, 可选) 在训练期间冻结前 n 层，或冻结层索引列表。
            half=False,
            # 分割
            overlap_mask=True,  # (bool) 训练期间是否应重叠掩码（仅适用于分割训练）
            mask_ratio=4,  # (int) 掩码降采样比例（仅适用于分割训练）
            # 分类
            dropout=0.0,  # (float) 使用丢弃正则化（仅适用于分类训练）
            # 超参数 ----------------------------------------------------------------------------------------------
            lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
            lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
            momentum=0.937,  # (float) SGD动量/Adam beta1
            weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
            warmup_epochs=3.0,  # (float) 预热周期（分数可用）
            warmup_momentum=0.8,  # (float) 预热初始动量
            warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
            box=7.5,  # (float) 盒损失增益
            cls=0.5,  # (float) 类别损失增益（与像素比例）
            dfl=1.5,  # (float) dfl损失增益
            pose=12.0,  # (float) 姿势损失增益
            kobj=1.0,  # (float) 关键点对象损失增益
            label_smoothing=0.0,  # (float) 标签平滑（分数）
            nbs=64,  # (int) 名义批量大小
            hsv_h=0.015,  # (float) 图像HSV-Hue增强（分数）
            hsv_s=0.7,  # (float) 图像HSV-Saturation增强（分数）
            hsv_v=0.4,  # (float) 图像HSV-Value增强（分数）
            degrees=0.0,  # (float) 图像旋转（+/- deg）
            translate=0.1,  # (float) 图像平移（+/- 分数）
            scale=0.5,  # (float) 图像缩放（+/- 增益）
            shear=0.0,  # (float) 图像剪切（+/- deg）
            perspective=0.0,  # (float) 图像透视（+/- 分数），范围为0-0.001
            flipud=0.0,  # (float) 图像上下翻转（概率）
            fliplr=0.5,  # (float) 图像左右翻转（概率）
            mosaic=1.0,  # (float) 图像马赛克（概率）
            mixup=0.0,  # (float) 图像混合（概率）
            copy_paste=0.0,  # (float) 分割复制-粘贴（概率）
            time=None,  # (float, 可选) 训练的时间上限，单位是小时。如果设置，会覆盖epochs参数
            multi_scale=False,  # (bool) 是否使用多尺度训练，能够增强模型对不同图像尺寸的适应能力
            copy_paste_mode="flip",  # (str) 复制粘贴增强的模式，可以是'flip'（翻转）或'mixup'（混合），用于数据增强
            auto_augment="randaugment",  # (str) 自动增强策略，选择如'randaugment'，用于分类任务的数据增强
            erasing=0.4,  # (float) 随机擦除的概率，用于分类训练中的数据增强，0表示不进行擦除
            crop_fraction=1.0,  # (float) 图像裁剪的比例，用于分类训练，1.0表示不裁剪，范围应大于0且不超过1
        )
        print(f"模型 {config['yaml_path']} 训练结束，输出目录：runs/train/{config['exp_name']}\n")

def main():
    train_multiple_models()

if __name__ == '__main__':
    main()
