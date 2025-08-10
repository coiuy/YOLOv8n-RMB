import albumentations as A
import cv2

COMPLEX_STRATEGY = A.Compose(
    [
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=10, p=0.3),
        # 安全裁剪 + 随机裁剪
        A.BBoxSafeRandomCrop(p=0.2),

        # 颜色 / 光照
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # 模拟天气
        A.RandomRain(p=0.1),
        A.RandomFog(p=0.1),

        # 模糊 & 噪声
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Cutout
        A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        min_area=5,
        min_visibility=0.2,
        label_fields=['class_labels']
    )
)


