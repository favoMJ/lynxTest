import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model.vovnet_ import VoVNet
from model.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion
import math

class DBGAMoudle(nn.Module):
    def __init__(self,in_chan1,in_chan2,out_chan,norm_layer=nn.GroupNorm):
        super(DBGAMoudle, self).__init__()
        self.conv_x1 = nn.Conv2d(in_chan1,
                out_chan,
                kernel_size = 3,
                stride = 1,
                padding = 1)
        self.conv_x2 = nn.Conv2d(in_chan2,
                out_chan,
                kernel_size = 3,
                stride = 1,
                padding = 1)
        self.sigmoid = nn.Sigmoid()
        self.convx = ConvBnRelu(out_chan, out_chan, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self,x1,x2):
        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)
        x2 = self.sigmoid(x2)
        temp_x1 = x1
        x1 = x1 * x2
        x2 = x2 * temp_x1
        x = x1 + x2
        x = self.convx(x)
        return x


def conv2d_sample_by_sample(
    x: torch.Tensor,
    weight: torch.Tensor,
    oup: int,
    inp: int,
    ksize: int,
    stride: int,
    groups: int,
) -> torch.Tensor:
    padding, batch_size = ksize // 2, x.shape[0]
    if batch_size == 1:
        out = F.conv2d(
            x,
            weight=weight.view(oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups,
        )
    else:
        out = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            weight.view(batch_size * oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups * batch_size,
        )
        out = out.permute([1, 0, 2, 3]).view(
            batch_size, oup, out.shape[2], out.shape[3]
        )
    return out

class WeightNet(nn.Module):
    r"""Applies WeightNet to a standard convolution.

    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.

    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride, ratio = 2):
        super().__init__()

        self.M = 2
        self.G = 2

        self.ratio = ratio
        self.pad = ksize // 2
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = ConvBnRelu(inp_gap, self.M * oup, 1, 1, 0,
                   has_bn=False,
                   has_relu=False, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = ConvBnRelu(self.M * oup, oup * inp * ksize * ksize, 1, 1, 0,
                                 groups=self.G * oup,
                                 has_bn=False,
                                 has_relu=False, has_bias=False)


    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.oup, self.inp, self.ksize, self.stride, 1
        )
##

class FFENet(nn.Module):
    def __init__(self, classes, is_training=True,
                 pretrained_model=None,
                 norm_layer=nn.GroupNorm):
        super(FFENet, self).__init__()
        self.context_path = VoVNet("vovnet-19-slim", num_classes=classes)

        self.business_layer = []
        self.is_training = is_training

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )


        arms = [WeightNet(512, conv_channel,1,1),
                WeightNet(384, conv_channel,1,1)]


        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]


        heads = [FFENetHead(conv_channel, classes, 16,
                             True, norm_layer),
                 FFENetHead(conv_channel, classes, 8,
                             True, norm_layer),
                 FFENetHead(64, classes, 4,
                             False, norm_layer)]

        reduces = [ConvBnRelu(512, max(16, 512//16), 1, 1, 0,
                         has_bn=True,
                         has_relu=True, has_bias=False, norm_layer=norm_layer),
                   ConvBnRelu(384, max(16, 384 // 16), 1, 1, 0,
                            has_bn=True,
                            has_relu=True, has_bias=False, norm_layer=norm_layer)]


        self.ffm =  DBGAMoudle(128, 128, 112)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)
        self.reduces = nn.ModuleList(reduces)

        self.spatial = ConvBnRelu(256, conv_channel, 1, 1, 0,
                   has_bn=True,
                   has_relu=True, has_bias=False, norm_layer=norm_layer)
        self.DAP = nn.Sequential(
            nn.PixelShuffle(2),
           nn.AvgPool2d((2,2))
        )
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)
        self.business_layer.append(self.reduces)


    def forward(self, data, label=None):
        context_blocks = self.context_path(data)
        spatial_out = context_blocks[1]
        spatial_out = self.spatial(spatial_out)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine,reduce) in enumerate(zip(context_blocks[:3], self.arms,
                                                  self.refines,self.reduces)):
            x_gap = fm.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)
            x_gap = reduce(x_gap)
            fm = arm(fm,x_gap)
            fm += last_fm

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            pred_out.append(last_fm)
            last_fm = refine(last_fm)

        context_out = last_fm
#132.06


        concate_fm = self.ffm(spatial_out, context_out)
        last_fm = F.interpolate(concate_fm, size=(context_blocks[-1].size()[2:]),
                                mode='bilinear', align_corners=True)
        concate_fm = last_fm + torch.sigmoid(context_blocks[-1])*context_blocks[-1]

        pred_out.append(concate_fm)
        out = []
        out1 = self.heads[-1](pred_out[-1])
        out.append(out1)
        if self.is_training:
            out2 = self.heads[0](pred_out[0])
            out3 = self.heads[1](pred_out[1])

            out2=F.interpolate(out2, size=(out1.size()[2:]),
                          mode='bilinear', align_corners=True)
            out3=F.interpolate(out3, size=(out1.size()[2:]),
                          mode='bilinear', align_corners=True)
            out.append(out2)
            out.append(out3)

        return out



class FFENetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.GroupNorm):
        super(FFENetHead, self).__init__()
        self.is_aux = is_aux
        if is_aux:
            self.conv_3x3 = ConvBnRelu(32, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(28, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
        self.DAP =  nn.Sequential(nn.PixelShuffle(2), nn.AvgPool2d((2,2)))
    def forward(self, x):
        x = self.DAP(x)
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)

        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output

from torchsummary import summary
if __name__ == "__main__":
    model = FFENet(11)
    model.cuda()
    model.eval()
    summary(model,(3,360,480))

