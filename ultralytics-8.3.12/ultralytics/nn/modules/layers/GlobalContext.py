# 论文名称：GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
# 论文地址：https://arxiv.org/pdf/1904.11492.pdf
# 代码地址：https://github.com/xvjiarui/GCNet
# 博客地址：https://yolov5.blog.csdn.net/article/details/130644184

import torch
from torch import nn as nn
import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d

__all__ = ["GlobalContext", "C3_GlobalContext", "C2f_GlobalContext"]


class GlobalContext(nn.Module):
    def __init__(
        self,
        channels,
        use_attn=True,
        fuse_add=False,
        fuse_scale=True,
        init_last_zero=False,
        rd_ratio=1.0 / 8,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()

        # 检查 channels 是否有效
        assert (
            isinstance(channels, int) and channels > 0
        ), f"channels 必须是正整数，当前值为: {channels}"

        # 获取激活层
        act_layer = get_act_layer(act_layer)

        # 生成 self.conv_attn 层，如果 use_attn 为 True，则初始化 conv_attn
        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        # 计算 rd_channels，如果为 None，则计算默认值
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )

        # 检查 rd_channels 的有效性
        assert (
            isinstance(rd_channels, int) and rd_channels > 0
        ), f"rd_channels 必须是正整数，当前值为: {rd_channels}"

        # 选择性创建 mlp_add 层
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None

        # 选择性创建 mlp_scale 层
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        # 创建激活层
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        # 检查输入是否符合期望的形状
        assert isinstance(
            x, torch.Tensor
        ), f"x 必须是 torch.Tensor 类型，当前类型为: {type(x)}"
        assert x.dim() == 4, f"x 的维度必须为 4，当前维度为: {x.dim()}"

        # 计算注意力分数
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        # 对输入应用缩放和加法
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GlobalC_Bottleneck(nn.Module):
    #  Bottleneck with 1 Attention
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.att = GlobalContext(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.att(self.cv2(self.cv1(x)))
            if self.add
            else self.att(self.cv2(self.cv1(x)))
        )


class C3_GlobalContext(nn.Module):
    # CSP Bottleneck with 3 convolutions and 1 GlobalContext. by CSDN迪菲赫尔曼
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(GlobalC_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class GlobalContext_Bottleneck_(nn.Module):
    def __init__(
        self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.effse = GlobalContext(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.effse(self.cv2(self.cv1(x)))
            if self.add
            else self.effse(self.cv2(self.cv1(x)))
        )


class C2f_GlobalContext(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 GlobalContext. by csdn迪菲赫尔曼"""

    def __init__(
        self, c1, c2, n=2, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        assert n > 0, "n 必须大于 0"
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            GlobalContext_Bottleneck_(
                self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for idx, m in enumerate(self.m):
            y.append(m(y[-1]))
        concatenated_shape = torch.cat(y, 1).shape
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
