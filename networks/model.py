# @Time     : 2022/4/14 20:39
# @Author   : Chen nengzhen
# @FileName : model.py
# @Software : PyCharm
import math
import torch.nn.functional as F
import torch
from torch import nn


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def convbn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_reflectance=True):
        super(Conv3x3, self).__init__()
        if use_reflectance:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AvgPool2d(1),  # 将nn.AdaptiveAvgPool2d(1)替换为AvgPool2d()
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )]
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            # 注意：如果改变了空洞卷积的数量，那么下面的卷积操作的输入通道数也要改变。
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class IDNet(nn.Module):
    def __init__(self):
        super(IDNet, self).__init__()

        """ RGB Encoder """

        self.rgb_conv_init = convbnrelu(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1_1 = BasicBlock(inplanes=32, planes=64, stride=2)
        self.rgb_encoder_layer1_2 = BasicBlock(inplanes=64, planes=64, stride=1)  # 1/2

        self.rgb_encoder_layer2_1 = BasicBlock(inplanes=64, planes=128, stride=2)
        self.rgb_encoder_layer2_2 = BasicBlock(inplanes=128, planes=128, stride=1)  # 1/4

        self.rgb_encoder_layer3_1 = BasicBlock(inplanes=128, planes=256, stride=2)
        self.rgb_encoder_layer3_2 = BasicBlock(inplanes=256, planes=256, stride=1)  # 1/8

        self.rgb_encoder_layer4_1 = BasicBlock(inplanes=256, planes=512, stride=2)
        self.rgb_encoder_layer4_2 = BasicBlock(inplanes=512, planes=512, stride=1)  # 1/16

        self.rgb_encoder_layer5_1 = BasicBlock(inplanes=512, planes=1024, stride=2)
        self.rgb_encoder_layer5_2 = BasicBlock(inplanes=1024, planes=1024, stride=1)  # 1/32

        """ RGB Decoder """
        self.rgb_decoder_layer5 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)  # [bs, 512, 11, 38]
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)  # [bs, 256, 22, 76]
        self.rgb_decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)  # [bs, 128, 44, 152]
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)  # [bs, 64, 88, 304]
        self.rgb_decoder_layer1 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)  # [bs, 32, 176, 608]
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)  # [bs, 2, 352, 1216]

        self.convs_rgb_4to1 = Conv3x3(in_channels=128, out_channels=2)
        self.convs_rgb_2to1 = Conv3x3(in_channels=64, out_channels=2)

        """共享特征模块"""

        self.shared_conv0 = nn.Sequential(
            ASPP(in_channels=64, out_channels=32, atrous_rates=[1, 2, 4]),
        )
        self.aspp0_1 = conv3x3(in_planes=32, out_planes=64, stride=2)  # [1, 64, 176, 608]

        self.shared_conv1 = nn.Sequential(
            ASPP(in_channels=192, out_channels=64, atrous_rates=[1, 2, 4]),
        )
        self.aspp1_2 = conv3x3(in_planes=64, out_planes=128, stride=2)

        self.shared_conv2 = nn.Sequential(
            ASPP(in_channels=384, out_channels=128, atrous_rates=[1, 2, 4]),
        )
        self.aspp2_3 = conv3x3(in_planes=128, out_planes=256, stride=2)

        self.shared_conv3 = nn.Sequential(
            ASPP(in_channels=768, out_channels=256, atrous_rates=[1, 2, 4]),
        )
        self.aspp3_4 = conv3x3(in_planes=256, out_planes=512, stride=2)

        self.shared_conv4 = nn.Sequential(
            ASPP(in_channels=1536, out_channels=512, atrous_rates=[1, 2, 4]),
        )
        self.aspp4_5 = conv3x3(in_planes=512, out_planes=1024, stride=2)

        self.shared_conv5 = nn.Sequential(
            ASPP(in_channels=3072, out_channels=1024, atrous_rates=[1, 2, 4]),
        )

        """  Image Reconstruction  """
        self.reconstruction4 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.reconstruction3 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.reconstruction2 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.reconstruction1 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.reconstruction0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.reconstruction_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)  # 输出重构gray image

        """ Depth Encoder"""
        self.depth_conv_init = convbnrelu(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_encoder_layer1_1 = BasicBlock(inplanes=32, planes=64, stride=2)
        self.depth_encoder_layer1_2 = BasicBlock(inplanes=64, planes=64, stride=1)

        self.depth_encoder_layer2_1 = BasicBlock(inplanes=64, planes=128, stride=2)
        self.depth_encoder_layer2_2 = BasicBlock(inplanes=128, planes=128, stride=1)

        self.depth_encoder_layer3_1 = BasicBlock(inplanes=128, planes=256, stride=2)
        self.depth_encoder_layer3_2 = BasicBlock(inplanes=256, planes=256, stride=1)

        self.depth_encoder_layer4_1 = BasicBlock(inplanes=256, planes=512, stride=2)
        self.depth_encoder_layer4_2 = BasicBlock(inplanes=512, planes=512, stride=1)

        self.depth_encoder_layer5_1 = BasicBlock(inplanes=512, planes=1024, stride=2)
        self.depth_encoder_layer5_2 = BasicBlock(inplanes=1024, planes=1024, stride=1)

        """ Depth Decoder"""
        self.depth_decoder_layer5 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_layer4 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.depth_decoder_layer1 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.depth_decoder_output = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, padding=1)

        self.convs_depth_4to1 = Conv3x3(in_channels=128, out_channels=2)
        self.convs_depth_2to1 = Conv3x3(in_channels=64, out_channels=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb = x["rgb"]
        depth = x["d"]
        """ Get RGB Encoder and Depth Encoder Feature Maps """
        features_rgb_layer0 = self.rgb_conv_init(rgb)  # 32, 1/1
        features_depth_layer0 = self.depth_conv_init(depth)  # 32, 1/1
        shared_conv0_out = self.shared_conv0(torch.cat([features_rgb_layer0, features_depth_layer0], dim=1))  # 32, 1/1
        # print("shared_conv0_out.shape: ", shared_conv0_out.shape)  # [1, 32, 352, 1216]
        f0_1 = self.aspp0_1(shared_conv0_out)

        features_rgb_layer1 = self.rgb_encoder_layer1_2(self.rgb_encoder_layer1_1(features_rgb_layer0 + shared_conv0_out))  # 64, 1/2
        features_depth_layer1 = self.depth_encoder_layer1_2(self.depth_encoder_layer1_1(features_depth_layer0 + shared_conv0_out))  # 64, 1/2
        shared_conv1_out = self.shared_conv1(torch.cat([f0_1, features_rgb_layer1, features_depth_layer1], dim=1))
        # print("shared_conv1_out.shape: ", shared_conv1_out.shape)  # [1, 64, 176, 608]
        f1_2 = self.aspp1_2(shared_conv1_out)

        features_rgb_layer2 = self.rgb_encoder_layer2_2(self.rgb_encoder_layer2_1(features_rgb_layer1 + shared_conv1_out))  # 128, 1/4
        features_depth_layer2 = self.depth_encoder_layer2_2(self.depth_encoder_layer2_1(features_depth_layer1 + shared_conv1_out))  # 128, 1/4
        shared_conv2_out = self.shared_conv2(torch.cat([f1_2, features_rgb_layer2, features_depth_layer2], dim=1))
        # print("shared_conv2_out.shape: ", shared_conv2_out.shape)  # [1, 128, 88, 304]
        f2_3 = self.aspp2_3(shared_conv2_out)

        features_rgb_layer3 = self.rgb_encoder_layer3_2(self.rgb_encoder_layer3_1(features_rgb_layer2 + shared_conv2_out))  # 256, 1/8
        features_depth_layer3 = self.depth_encoder_layer3_2(self.depth_encoder_layer3_1(features_depth_layer2 + shared_conv2_out))  # 256, 1/8
        shared_conv3_out = self.shared_conv3(torch.cat([f2_3, features_rgb_layer3, features_depth_layer3], dim=1))
        # print("shared_conv3_out.shape: ", shared_conv3_out.shape)  # [1, 256, 44, 152]
        f3_4 = self.aspp3_4(shared_conv3_out)

        features_rgb_layer4 = self.rgb_encoder_layer4_2(self.rgb_encoder_layer4_1(features_rgb_layer3 + shared_conv3_out))  # 512, 1/16
        features_depth_layer4 = self.depth_encoder_layer4_2(self.depth_encoder_layer4_1(features_depth_layer3 + shared_conv3_out))  # 512, 1/16
        shared_conv4_out = self.shared_conv4(torch.cat([f3_4, features_rgb_layer4, features_depth_layer4], dim=1))  # 512, 1/16
        # print("shared_conv4_out.shape: ", shared_conv4_out.shape)  # [1, 512, 222, 76]
        f4_5 = self.aspp4_5(shared_conv4_out)

        features_rgb_layer5 = self.rgb_encoder_layer5_2(self.rgb_encoder_layer5_1(features_rgb_layer4 + shared_conv4_out))  # 1024, 1/32
        features_depth_layer5 = self.depth_encoder_layer5_2(self.depth_encoder_layer5_1(features_depth_layer4 + shared_conv4_out))  # 1024, 1/32
        shared_conv5_out = self.shared_conv5(torch.cat([f4_5, features_rgb_layer5, features_depth_layer5], dim=1))
        # print("shared_conv5_out.shape: ", shared_conv5_out.shape)   # [1, 1024, 11, 38]

        # ########################### Gray Image Reconstruction ###################
        reconstruction_4 = self.reconstruction4(shared_conv5_out)
        reconstruction_3 = self.reconstruction3(reconstruction_4)
        reconstruction_2 = self.reconstruction2(reconstruction_3)
        reconstruction_1 = self.reconstruction1(reconstruction_2)
        reconstruction_0 = self.reconstruction0(reconstruction_1)
        reconstruction_gray_image = self.reconstruction_output(reconstruction_0)

        """ RGB Decoder """
        r_depth5 = self.rgb_decoder_layer5(features_rgb_layer5 + shared_conv5_out)  # [bs, 512, 22, 76]
        r_depth4 = self.rgb_decoder_layer4(shared_conv4_out + r_depth5)  # [bs, 256, 44, 152]
        r_depth3 = self.rgb_decoder_layer3(shared_conv3_out + r_depth4)  # [bs, 128, 88, 304]
        r_depth2 = self.rgb_decoder_layer2(shared_conv2_out + r_depth3)  # [bs, 64, 176, 608]
        r_depth1 = self.rgb_decoder_layer1(shared_conv1_out + r_depth2)  # [bs, 32, 352, 1216]
        r_depth0 = self.rgb_decoder_output(features_rgb_layer0 + r_depth1)

        """ Depth Decoder """
        d_depth5 = self.depth_decoder_layer5(features_depth_layer5 + shared_conv5_out)
        d_depth4 = self.depth_decoder_layer4(shared_conv4_out + d_depth5)
        d_depth3 = self.depth_decoder_layer3(shared_conv3_out + d_depth4)  # [b, 128, 88, 304]
        d_depth2 = self.depth_decoder_layer2(shared_conv2_out + d_depth3)  # [b, 64, 176, 608]
        d_depth1 = self.depth_decoder_layer1(shared_conv1_out + d_depth2)  # [b, 32, 352, 1216]
        d_depth0 = self.depth_decoder_output(features_depth_layer0 + d_depth1)   # [b, 2, 352, 1216]

        scale_rgb_depth_4_1, r_conf_4_1 = torch.chunk(self.convs_rgb_4to1(r_depth3), chunks=2, dim=1)
        scale_rgb_depth_2_1, r_conf_2_1 = torch.chunk(self.convs_rgb_2to1(r_depth2), chunks=2, dim=1)
        scale_rgb_depth_1_1, r_conf_1_1 = torch.chunk(r_depth0, chunks=2, dim=1)

        scale_d_depth_4_1, d_conf_4_1 = torch.chunk(self.convs_depth_4to1(d_depth3), chunks=2, dim=1)
        scale_d_depth_2_1, d_conf_2_1 = torch.chunk(self.convs_depth_2to1(d_depth2), chunks=2, dim=1)
        scale_d_depth_1_1, d_conf_1_1 = torch.chunk(d_depth0, chunks=2, dim=1)

        r_conf_4_1, d_conf_4_1 = torch.chunk(self.softmax(torch.cat((r_conf_4_1, d_conf_4_1), dim=1)), chunks=2, dim=1)
        r_conf_2_1, d_conf_2_1 = torch.chunk(self.softmax(torch.cat((r_conf_2_1, d_conf_2_1), dim=1)), chunks=2, dim=1)
        r_conf_1_1, d_conf_1_1 = torch.chunk(self.softmax(torch.cat((r_conf_1_1, d_conf_1_1), dim=1)), chunks=2, dim=1)

        fuse_depth_4_1 = r_conf_4_1 * scale_rgb_depth_4_1 + d_conf_4_1 * scale_d_depth_4_1
        fuse_depth_2_1 = r_conf_2_1 * scale_rgb_depth_2_1 + d_conf_2_1 * scale_d_depth_2_1
        fuse_depth_1_1 = r_conf_1_1 * scale_rgb_depth_1_1 + d_conf_1_1 * scale_d_depth_1_1

        fuse_depths = [fuse_depth_1_1, fuse_depth_2_1, fuse_depth_4_1]
        r_depths = [scale_rgb_depth_1_1, scale_rgb_depth_2_1, scale_rgb_depth_4_1]
        d_depths = [scale_d_depth_1_1, scale_d_depth_2_1, scale_d_depth_4_1]

        return fuse_depths, r_depths, d_depths, reconstruction_gray_image


if __name__ == '__main__':
    import numpy as np
    from dataloader.kitti_loader import KittiDepth
    from torch.utils.data import DataLoader
    from options import args
    from thop import profile

    net = IDNet()
    num_params = sum([np.prod(p.size()) for p in net.parameters()])
    # print(num_params / 1000000.0, "M")
    print("params: %.2f M" % (num_params / 1000000.0))
    num_params_update = sum([np.prod(p.shape) for p in net.parameters() if p.requires_grad])
    # print(num_params_update / 1000000.0, "M")
    print("params learnable: {:.2f} M".format(num_params_update / 1000000.0))
    #
    dataset = KittiDepth(split="val", args=args)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for idx, item in enumerate(loader):
        # flops, params = profile(net, (item,))
        # print("flops: ", flops, " params: ", params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        f, r, d, recons = net(item)
        print(f[0].shape, r[0].shape, d[0].shape, recons.shape)
        break

