import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .blocks import Mlp
from ....builder import BACKBONES


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# ECA模块
class ECA_Layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA_Layer, self).__init__()  # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)  # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()  # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        return x * y.expand_as(x)  # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算


class ccw_body(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super(ccw_body, self).__init__()
        self.conv_1 = nn.Sequential(
            self.depthwise_conv(in_dim, in_dim, kernel_size=5, stride=1, padding=1),
            nn.Conv2d(in_dim, 9, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.caLayer_1 = SALayer(kernel_size=5, bias=False)
        self.down_1 = nn.Conv2d(9, 36, kernel_size=1, stride=2, padding=0)

        self.conv_2 = nn.Sequential(
            self.depthwise_conv(36, 36, kernel_size=5, stride=1, padding=1),
            nn.Conv2d(36, 36, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)
        self.caLayer_2 = SALayer(kernel_size=5, bias=False)
        self.down_2 = nn.Conv2d(36, 36, kernel_size=1, stride=2, padding=0)

        self.conv_3 = nn.Sequential(
            self.depthwise_conv(36, 36, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(36, out_dim, kernel_size=1, stride=1, padding=0, bias=False), )
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)
        self.saLayer_3 = ECA_Layer(channel=36)
        self.down_3 = nn.MaxPool2d(2)

        self.conv_4 = nn.Sequential(
            self.depthwise_conv(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False), )
        self.act_4 = nn.LeakyReLU(0.2, inplace=True)
        self.saLayer_4 = ECA_Layer(channel=out_dim)
        self.down_4 = nn.MaxPool2d(2)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, img):
        img = self.conv_1(img)
        img = self.act_1(img)
        img = self.down_1(img)
        img = self.caLayer_1(img) + img

        img = self.conv_2(img)
        img = self.act_2(img)
        img = self.down_2(img)
        img = self.caLayer_2(img) + img

        img = self.conv_3(img)
        img = self.act_3(img)
        img = self.saLayer_3(img) + img
        img = self.down_3(img)

        img = self.conv_4(img)
        img = self.act_4(img)
        img = self.saLayer_4(img) + img
        img = self.down_4(img)

        return img


class ccm_body(nn.Module):
    def __init__(self, in_dim=None, out_dim=None):
        super(ccm_body, self).__init__()
        self.conv_1 = nn.Sequential(
            self.depthwise_conv(in_dim, in_dim, kernel_size=5, stride=1, padding=1),
            nn.Conv2d(in_dim, 9, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.down_1 = nn.Conv2d(9, 36, kernel_size=1, stride=2, padding=0)
        self.caLayer_1 = SALayer(kernel_size=5, bias=False)

        self.conv_2 = nn.Sequential(
            self.depthwise_conv(36, 36, kernel_size=5, stride=1, padding=1),
            nn.Conv2d(36, 36, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)
        self.down_2 = nn.Conv2d(36, 36, kernel_size=1, stride=2, padding=0)
        self.caLayer_2 = SALayer(kernel_size=5, bias=False)

        self.conv_3 = nn.Sequential(
            self.depthwise_conv(36, 36, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(36, out_dim, kernel_size=1, stride=1, padding=0, bias=False), )
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)
        self.saLayer_3 = ECA_Layer(channel=36)
        self.down_3 = nn.MaxPool2d(2)

        self.conv_4 = nn.Sequential(
            self.depthwise_conv(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False), )
        self.act_4 = nn.LeakyReLU(0.2, inplace=True)
        self.saLayer_4 = ECA_Layer(channel=out_dim)
        self.down_4 = nn.MaxPool2d(2)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, img):
        img = self.conv_1(img)
        img = self.act_1(img)
        img = self.down_1(img)
        img = self.caLayer_1(img) + img

        img = self.conv_2(img)
        img = self.act_2(img)
        img = self.down_2(img)
        img = self.caLayer_2(img) + img

        img = self.conv_3(img)
        img = self.act_3(img)
        img = self.saLayer_3(img) + img
        img = self.down_3(img)

        img = self.conv_4(img)
        img = self.act_4(img)
        img = self.saLayer_4(img) + img
        img = self.down_4(img)

        return img


class Light_isp_ccm(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super(Light_isp_ccm, self).__init__()
        self.isp_body = ccm_body(in_dim=in_dim, out_dim=out_dim)
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim * 2 * 2, hidden_features=out_dim, out_features=9)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccm_base = nn.Parameter(torch.eye((3)), requires_grad=True)
        
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img):
        out = self.isp_body(img)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = img.shape[0]
        ccm = torch.reshape(out, shape=(b, 3, 3)) + self.ccm_base
        img = img.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccm[i, :, :]) for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
        return img


class Light_isp_ccw_gamma(nn.Module):
    def __init__(self, in_dim=3, out_dim=36):
        super(Light_isp_ccw_gamma, self).__init__()
        self.isp_body = ccw_body(in_dim=in_dim, out_dim=out_dim)
        self.adAvp = nn.AdaptiveMaxPool2d(2)
        self.mlp_1 = Mlp(in_features=out_dim * 2 * 2, hidden_features=out_dim, out_features=4)
        self.flat_1 = nn.Flatten(start_dim=1)
        self.ccw_base = nn.Parameter(torch.eye((3)), requires_grad=True)
        self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img):
        out = self.isp_body(img)
        out = self.adAvp(out)
        out = self.flat_1(out)
        out = self.mlp_1(out)
        b = img.shape[0]
        gamma = out[:, 0:1] + self.gamma_base
        ccw = torch.reshape(torch.diag_embed(out[:, 1:]), shape=(b, 3, 3)) + self.ccw_base
        img = img.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        img = torch.stack([self.apply_color(img[i, :, :, :], ccw[i, :, :]) ** gamma[i, :] for i in range(b)], dim=0)
        img = img.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

        return img


class CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # activated_layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class res_CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # activated_layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + x


class DecomNet(nn.Module):
    def __init__(self, in_channels,mid_channels, out_channels, kernel_size):
        super().__init__()

        # init conv
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride=1, padding=1, groups=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act2 = nn.Tanh()
        self.act3 = nn.Tanh()

        # activated_layer
        self.cba1 = CBA(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size)
        self.cba2 = res_CBA(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size)
        self.cba3 = res_CBA(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size)
        self.cba4 = res_CBA(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size)

        # recon_layer
        self.conv_recon = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.cba1(x)
        x = self.cba2(x)
        x = self.cba3(x)
        x = self.conv_recon(x)
        R_output = self.act2(x[:, 0:3, :, :])
        L_output = self.act3(x[:, 3:6, :, :])

        return R_output, L_output


@BACKBONES.register_module()
class RetinexNet(nn.Module):
    def __init__(self, ):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet(in_channels=3, mid_channels=18, out_channels=6, kernel_size=3)
        self.ccm = Light_isp_ccm(in_dim=3)
        self.ccw_gamma = Light_isp_ccw_gamma(in_dim=3)
        self.CBA1 = CBA(in_channels=3, out_channels=18, kernel_size=3)
        self.CBA2 = CBA(in_channels=18, out_channels=18, kernel_size=3)
        self.CBA3 = CBA(in_channels=18, out_channels=18, kernel_size=3)
        self.CBA4 = CBA(in_channels=18, out_channels=3, kernel_size=3)

    def forward(self, img_low):
        R_output, L_output = self.DecomNet(img_low)
        R_restore_img = self.ccm(R_output)
        L_restore_img = self.ccw_gamma(L_output)
        output = self.CBA1(R_restore_img * L_restore_img)
        output = self.CBA2(output)
        output = self.CBA3(output)
        output = self.CBA4(output)
        return output, output, output


if __name__ == '__main__':
    img = torch.Tensor(2, 3, 400, 600)
    # net = DecomNet(in_channels=3, mid_channels=18,out_channels=6, kernel_size=3)
    net = RetinexNet()
    print('total parameters:', sum(param.numel() for param in net.parameters()))  # 91154
    # a = net(img)
    # output, _, _ = net(img)
    # print(output.shape)
