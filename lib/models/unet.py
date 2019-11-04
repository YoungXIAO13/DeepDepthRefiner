import torch.nn as nn
import torch
from .basic_modules import ConvBnRelu, ConvBnLeakyRelu, RefineResidual


class UNet(nn.Module):
    def __init__(self, depth_channels=1, occ_channels=9):
        super(UNet, self).__init__()

        self.down_scale = nn.MaxPool2d(2)
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.depth_down_layer0 = ConvBnLeakyRelu(depth_channels, 32, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer1 = ConvBnLeakyRelu(32, 64, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer2 = ConvBnLeakyRelu(64, 128, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer3 = ConvBnLeakyRelu(128, 256, 3, 1, 1, 1, 1,\
                                     has_bn=True, leaky_alpha=0.3, \
                                     has_leaky_relu=True, inplace=True, has_bias=True)

        self.occ_down_layer0 = ConvBnRelu(occ_channels, 32, 3, 1, 1, 1, 1,
                                          has_bn=True,
                                          inplace=True, has_bias=True)
        self.occ_down_layer1 = ConvBnRelu(32, 64, 3, 1, 1, 1, 1,
                                          has_bn=True,
                                          inplace=True, has_bias=True)
        self.occ_down_layer2 = ConvBnRelu(64, 128, 3, 1, 1, 1, 1,
                                          has_bn=True,
                                          inplace=True, has_bias=True)
        self.occ_down_layer3 = ConvBnRelu(128, 256, 3, 1, 1, 1, 1,
                                          has_bn=True,
                                          inplace=True, has_bias=True)

        self.depth_up_layer0 = RefineResidual(256 * 2, 128, relu_layer='LeakyReLU', \
                                     has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer1 = RefineResidual(128 * 3, 64, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer2 = RefineResidual(64 * 3, 32, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer3 = RefineResidual(32 * 3, 32, relu_layer='LeakyReLU', \
                                    has_bias=True, has_relu=True, leaky_alpha=0.3)

        self.refine_layer0 = ConvBnLeakyRelu(32 + occ_channels + depth_channels, 16, 3, 1, 1, 1, 1,\
                                    has_bn=True, leaky_alpha=0.3, \
                                    has_leaky_relu=True, inplace=True, has_bias=True)
        self.refine_layer1 = ConvBnLeakyRelu(16, 10, 3, 1, 1, 1, 1,\
                                    has_bn=True, leaky_alpha=0.3, \
                                    has_leaky_relu=True, inplace=True, has_bias=True)

        self.output_layer = ConvBnRelu(10, 1, 3, 1, 1, 1, 1,\
                                     has_bn=False, \
                                     has_relu=False, inplace=True, has_bias=True)

    def forward(self, occ, x):
        #### Occlusion ####
        r1 = self.occ_down_layer0(occ)
        r1 = self.down_scale(r1)
        r2 = self.occ_down_layer1(r1)
        r2 = self.down_scale(r2)
        r3 = self.occ_down_layer2(r2)
        r3 = self.down_scale(r3)
        r4 = self.occ_down_layer3(r3)
        r4 = self.down_scale(r4)

        #### Depth ####
        x1 = self.depth_down_layer0(x)
        x1 = self.down_scale(x1)
        x2 = self.depth_down_layer1(x1)
        x2 = self.down_scale(x2)
        x3 = self.depth_down_layer2(x2)
        x3 = self.down_scale(x3)
        x4 = self.depth_down_layer3(x3)
        x4 = self.down_scale(x4)

        #### Decode ####
        m4 = torch.cat((r4, x4), 1)
        m4 = self.depth_up_layer0(m4)
        m3 = self.up_scale(m4)

        m3 = torch.cat((m3, r3, x3), 1)
        m3 = self.depth_up_layer1(m3)
        m2 = self.up_scale(m3)

        m2 = torch.cat((m2, r2, x2), 1)
        m2 = self.depth_up_layer2(m2)
        m1 = self.up_scale(m2)

        m1 = torch.cat((m1, r1, x1), 1)
        m1 = self.depth_up_layer3(m1)
        m = self.up_scale(m1)

        x = torch.cat((m, occ, x), 1)
        x = self.refine_layer0(x)
        x = self.refine_layer1(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    model = UNet()

    depth = torch.rand((4, 1, 480, 640))
    occ = torch.rand((4, 9, 480, 640))

    out = model(occ, depth)
    print(out.shape)
