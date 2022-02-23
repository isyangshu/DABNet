# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES, build_backbone
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import ConvModule, Scale
from mmseg.utils import get_root_logger
from ..utils import ResLayer
from mmcv.runner import BaseModule
from einops import rearrange, repeat
from mmseg.ops import resize
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import NORM_LAYERS

class InputProjectionA(nn.Module):
    def __init__(self, samplingTimes):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(2, stride=2, padding=0))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class SIM(nn.Module):
    def __init__(self, h_C, l_C, norm_layer='BN2d'):
        super(SIM, self).__init__()

        norm_layer = NORM_LAYERS.get(norm_layer)

        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(h_C*4, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.bnl_0 = norm_layer(l_C)
        self.bnh_0 = norm_layer(l_C)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.l2l_p1 = nn.Conv2d(l_C, l_C, 1, 1, 0)
        self.l2l_p2 = nn.Conv2d(h_C, l_C, 1, 1, 0)
        self.l2l_bn = norm_layer(l_C)

        self.h2h_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = norm_layer(l_C)
        self.bnh_1 = norm_layer(l_C)

        self.h2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = norm_layer(h_C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        H, W = x.shape[2:]  # N, C, H, W
        x_patch_1 = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2)  # N, 4*C, H//2, W//2
        x_patch_2 = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w', h=2, w=2)  # N, C*H*W//4, 2, 2

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))  # N, C_, H, W
        x_l_1 = self.relu(self.bnl_0(self.h2l_0(x_patch_1)))  # N, C_, H, W
        x_l_2 = rearrange(self.avg(x_patch_2), 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=H//2, p2=W//2).contiguous()  # N, C, H//2, W//2
        x_l = self.relu(self.l2l_bn(self.l2l_p1(x_l_1)+self.l2l_p2(x_l_2)))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(H, W)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(H, W)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x

class FeatureFusionModule(nn.Module):
    """Feature Fusion Module to fuse low level output feature of Spatial Path
    and high level output feature of Context Path.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Feature Fusion Module.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_atten = nn.Sequential(
            # In paper, Gap+1*1 conv+relu+1*1 conv+sigmoid;
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Sigmoid())

    def forward(self, x_sp, x_cp):
        x_concat = torch.cat([x_sp, x_cp], dim=1)
        x_fuse = self.conv1(x_concat)
        x_atten = self.gap(x_fuse)
        x_atten = self.conv_atten(x_atten)
        x_atten = x_fuse * x_atten
        x_out = x_atten + x_fuse
        return x_out

class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module (ARM) to refine the features of each stage.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(AttentionRefinementModule, self).__init__()
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out

@BACKBONES.register_module()
class DABNet_PCE01(BaseModule):
    """BiSeNetV1 backbone.

    This backbone is the implementation of `BiSeNet: Bilateral
    Segmentation Network for Real-time Semantic
    Segmentation <https://arxiv.org/abs/1808.00897>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input
            image. Default: 3.
        spatial_channels (Tuple[int]): Size of channel numbers of
            various layers in Spatial Path.
            Default: (64, 64, 64, 128).
        context_channels (Tuple[int]): Size of channel numbers of
            various modules in Context Path.
            Default: (128, 256, 512).
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        out_channels(int): The number of channels of output.
            It must be the same with `in_channels` of decode_head.
            Default: 256.
    """

    def __init__(self,
                 backbone_cfg,
                 in_channels=3,
                 spatial_channels=(64, 64, 64, 128),
                 context_channels=(128, 256, 512),
                 out_indices=(0, 1, 2),
                 align_corners=False,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super(DABNet_PCE01, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone_cfg)

        self.out_indices = out_indices
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = dict(type='ReLU')
        norm_name = 'SyncBN'

        self.ffm = FeatureFusionModule(spatial_channels[3] * 2, out_channels, norm_cfg=self.norm_cfg)

        self.arm16 = AttentionRefinementModule(context_channels[1], spatial_channels[3], norm_cfg=self.norm_cfg)
        self.arm32 = AttentionRefinementModule(context_channels[2], spatial_channels[3], norm_cfg=self.norm_cfg)

        self.conv_head32 = ConvModule(
            in_channels=spatial_channels[3],
            out_channels=spatial_channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_head16 = ConvModule(
            in_channels=spatial_channels[3],
            out_channels=spatial_channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sp_0 = ConvModule(in_channels=spatial_channels[0] + 3,
                               out_channels=spatial_channels[1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        self.sp_1 = ConvModule(in_channels=spatial_channels[1],
                               out_channels=spatial_channels[2],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        self.sp_2 = ConvModule(in_channels=spatial_channels[2],
                               out_channels=spatial_channels[3],
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               conv_cfg=conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        self.SIM_0 = SIM(64, 32, norm_name)  # SyncBN
        self.SIM_1 = SIM(128, 64, norm_name)

        self.cp_1 = ConvModule(in_channels=spatial_channels[1],
                               out_channels=spatial_channels[1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        self.cp_2 = ConvModule(in_channels=context_channels[0],
                               out_channels=spatial_channels[2],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        self.sample1 = InputProjectionA(1)
    def forward(self, img):
        # stole refactoring code from Coin Cheung, thanks

        x_2 = self.backbone.stem(img)
        x_4 = self.backbone.maxpool(x_2)

        x_4 = self.backbone.layer1(x_4)
        x_4 = self.SIM_0(x_4)  # N， 64， 256， 256

        x_8 = self.backbone.layer2(x_4)  # N, 128, 128, 128
        x_8 = self.SIM_1(x_8)

        x_16 = self.backbone.layer3(x_8)  # N, 256, 64, 64

        x_32 = self.backbone.layer4(x_16)  # N, 512, 32, 32

        x_spatial = self.sample1(img)  # N, 3, 512, 512
        x_spatial = self.sp_0(torch.cat((x_spatial, x_2), dim=1))  # N, 64, 256, 256
        x_ = self.cp_1(x_4)
        x_spatial = self.sp_1(x_spatial+x_)  # N, 64, 128, 128
        x_ = self.cp_2(x_8)
        x_spatial = self.sp_2(x_spatial+x_)  # N, 128, 128, 128

        x_32_up = resize(input=self.arm32(x_32), size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)

        x_16_sum = self.arm16(x_16) + x_32_up
        x_16_up = resize(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)

        x_fuse = self.ffm(x_spatial, x_16_up)

        outs = [x_fuse, x_16_up, x_32_up]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
