""" Adapted from the original implementation. """
#FPS: 135.62
import collections
import dataclasses
from typing import List

import torch
import torch.nn as nn
from base_model.blurpool import BlurPool
import torch.nn.functional as F

@dataclasses.dataclass
class VoVNetParams:
    stem_out: int
    stage_conv_ch: List[int]  # Channel depth of
    stage_out_ch: List[int]  # The channel depth of the concatenated output
    layer_per_block: int
    block_per_stage: List[int]
    dw: bool


_STAGE_SPECS = {
    "vovnet-19-slim-dw": VoVNetParams(
        64, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], True
    ),
    "vovnet-19-dw": VoVNetParams(
        64, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], True
    ),
    "vovnet-19-slim": VoVNetParams(
        128, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], False
    ),
    "vovnet-19": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], False
    ),
    "vovnet-39": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 2, 2], False
    ),
    "vovnet-57": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 4, 3], False
    ),
    "vovnet-99": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 3, 9, 3], False
    ),
}

_BN_MOMENTUM = 1e-1
_BN_EPS = 1e-5


def dw_conv(
        in_channels: int, out_channels: int, stride: int = 1
) -> List[torch.nn.Module]:
    """ Depthwise separable pointwise linear convolution. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
            bias=False,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def conv(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
) -> List[torch.nn.Module]:
    """ 3x3 convolution with padding."""
    return [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def pointwise(in_channels: int, out_channels: int) -> List[torch.nn.Module]:
    """ Pointwise convolution."""
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


# As seen here: https://arxiv.org/pdf/1910.03151v4.pdf. Can outperform ESE with far fewer
# paramters.
class ESA(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        # BCHW -> BHCW
        y = y.permute(0, 2, 1, 3)
        y = self.conv(y)

        # Change the dimensions back to BCHW
        y = y.permute(0, 2, 1, 3)
        y = torch.sigmoid_(y)
        return x * y.expand_as(x)


class _OSA(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            layer_per_block: int,
            use_depthwise: bool = False,
    ) -> None:
        """ Implementation of an OSA layer which takes the output of its conv layers and
        concatenates them into one large tensor which is passed to the next layer. The
        goal with this concatenation is to preserve information flow through the model
        layers. This also ends up helping with small object detection.

        Args:
            in_channels: Channel depth of the input to the OSA block.
            stage_channels: Channel depth to reduce the input.
            concat_channels: Channel depth to force on the concatenated output of the
                comprising layers in a block.
            layer_per_block: The number of layers in this OSA block.
            use_depthwise: Wether to use depthwise separable pointwise linear convs.
        """
        super().__init__()
        # Keep track of the size of the final concatenation tensor.
        aggregated = in_channels
        self.isReduced = in_channels != stage_channels

        # If this OSA block is not the first in the OSA stage, we can
        # leverage the fact that subsequent OSA blocks have the same input and
        # output channel depth, concat_channels. This lets us reuse the concept of
        # a residual from ResNet models.
        self.identity = in_channels == concat_channels
        self.layers = torch.nn.ModuleList()
        self.use_depthwise = use_depthwise
        conv_op = dw_conv if use_depthwise else conv

        # If this model uses depthwise and the input channel depth needs to be reduced
        # to the stage_channels size, add a pointwise layer to adjust the depth. If the
        # model is not depthwise, let the first OSA layer do the resizing.
        if self.use_depthwise and self.isReduced:
            self.conv_reduction = torch.nn.Sequential(
                *pointwise(in_channels, stage_channels)
            )
            in_channels = stage_channels

        for _ in range(layer_per_block):
            self.layers.append(
                torch.nn.Sequential(*conv_op(in_channels, stage_channels))
            )
            in_channels = stage_channels

        # feature aggregation
        aggregated += layer_per_block * stage_channels
        self.concat = torch.nn.Sequential(*pointwise(aggregated, concat_channels))
        self.esa = ESA(concat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.identity:
            identity_feat = x

        output = [x]
        if self.use_depthwise and self.isReduced:
            x = self.conv_reduction(x)
        # Loop through all the
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.esa(xt)
        if self.identity:
            xt += identity_feat

        return xt


class _OSA_stage(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            block_per_stage: int,
            layer_per_block: int,
            stage_num: int,
            use_depthwise: bool = False,
    ) -> None:
        """An OSA stage which is comprised of OSA blocks.
        Args:
            in_channels: Channel depth of the input to the OSA stage.
            stage_channels: Channel depth to reduce the input of the block to.
            concat_channels: Channel depth to force on the concatenated output of the
                comprising layers in a block.
            block_per_stage: Number of OSA blocks in this stage.
            layer_per_block: The number of layers per OSA block.
            stage_num: The OSA stage index.
            use_depthwise: Wether to use depthwise separable pointwise linear convs.
        """
        super().__init__()

        # Use maxpool to downsample the input to this OSA stage.
        self.add_module(
            "Pooling", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        for idx in range(block_per_stage):
            # Add the OSA modules. If this is the first block in the stage, use the
            # proper in in channels, but the rest of the rest of the OSA layers will use
            # the concatenation channel depth outputted from the previous layer.
            self.add_module(
                f"OSA{stage_num}_{idx + 1}",
                _OSA(
                    in_channels if idx == 0 else concat_channels,
                    stage_channels,
                    concat_channels,
                    layer_per_block,
                    use_depthwise=use_depthwise,
                ),
            )


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(4,out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class VoVNet(torch.nn.Sequential):
    def __init__(
            self, model_name: str, num_classes: int = 10, input_channels: int = 3
    ) -> None:
        """
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.

        Usage:
        >>> net = VoVNet("vovnet-19-slim-dw", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-19-dw", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-19-slim", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-19", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-39", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-57", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])

        >>> net = VoVNet("vovnet-99", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        """
        super().__init__()
        assert model_name in _STAGE_SPECS, f"{model_name} not supported."

        stem_ch = _STAGE_SPECS[model_name].stem_out
        config_stage_ch = _STAGE_SPECS[model_name].stage_conv_ch
        config_concat_ch = _STAGE_SPECS[model_name].stage_out_ch
        block_per_stage = _STAGE_SPECS[model_name].block_per_stage
        layer_per_block = _STAGE_SPECS[model_name].layer_per_block
        conv_type = dw_conv if _STAGE_SPECS[model_name].dw else conv

        # Construct the stem.
        stem = conv(input_channels, 64, stride=2)
        stem += conv_type(64, 64)

        # The original implementation uses a stride=2 on the conv below, but in this
        # implementation we'll just pool at every OSA stage, unlike the original
        # which doesn't pool at the first OSA stage.
        stem += conv_type(64, stem_ch)
        self.model = torch.nn.Sequential()
        self.osa_0 = torch.nn.Sequential()
        self.osa_1 = torch.nn.Sequential()
        self.osa_2 = torch.nn.Sequential()
        self.osa_3 = torch.nn.Sequential()
        self.model.add_module("stem", torch.nn.Sequential(*stem))
        self._out_feature_channels = [stem_ch]

        # Organize the outputs of each OSA stage. This is the concatentated channel
        # depth of each sub block's layer's outputs.
        in_ch_list = [stem_ch] + config_concat_ch[:-1]

        '''
        class VoVNetParams:
        stem_out: int
        stage_conv_ch: List[int]  # Channel depth of
        stage_out_ch: List[int]  # The channel depth of the concatenated output
        layer_per_block: int
        block_per_stage: List[int]
        dw: bool
        ex:vovnet19 128, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], False
        '''

        # Add the OSA modules. Typically 4 modules.
        self.osa_0.add_module(f"OSA_{(0 + 2)}",_OSA_stage(
                    in_ch_list[0],
                    config_stage_ch[0],
                    config_concat_ch[0],
                    block_per_stage[0],
                    layer_per_block,
                    0 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[0])
        self.osa_1.add_module(f"OSA_{(1 + 2)}",_OSA_stage(
                    in_ch_list[1],
                    config_stage_ch[1],
                    config_concat_ch[1],
                    block_per_stage[1],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[1])
        self.osa_2.add_module(f"OSA_{(2 + 2)}",_OSA_stage(
                    in_ch_list[2],
                    config_stage_ch[2],
                    config_concat_ch[2],
                    block_per_stage[2],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[2])
        self.osa_3.add_module(f"OSA_{(3 + 2)}",_OSA_stage(
                    in_ch_list[3],
                    config_stage_ch[3],
                    config_concat_ch[3],
                    block_per_stage[3],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[3])
        # for idx in range(len(config_stage_ch)-2):
        #     print(idx)
        #     self.model.add_module(
        #         f"OSA_{(idx + 2)}",
        #         _OSA_stage(
        #             in_ch_list[idx],
        #             config_stage_ch[idx],
        #             config_concat_ch[idx],
        #             block_per_stage[idx],
        #             layer_per_block,
        #             idx + 2,
        #             _STAGE_SPECS[model_name].dw,
        #         ),
        #     )
        #
        #     self._out_feature_channels.append(config_concat_ch[idx])

        # Add the classification head.
        self.claaifier = torch.nn.BatchNorm2d(self._out_feature_channels[-1], _BN_MOMENTUM, _BN_EPS)

        # self.model.add_module(
        #     "classifier",
        #     torch.nn.Sequential(
        #         torch.nn.BatchNorm2d(
        #             self._out_feature_channels[-1], _BN_MOMENTUM, _BN_EPS
        #         ),
        #         # torch.nn.AdaptiveAvgPool2d(1),
        #         # torch.nn.Flatten(),
        #         # torch.nn.Dropout(0.2),
        #         # torch.nn.Linear(self._out_feature_channels[-1], num_classes, bias=True),
        #     ),
        # )

        norm_layer = nn.GroupNorm
        self.conv0 = ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        self.conv1 = ConvBnRelu(112, 112, 3, 1, 1,
                               has_bn=True, norm_layer=norm_layer,
                               has_relu=True, has_bias=False)
        self.conv2 = ConvBnRelu(256, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

        self.conv3 = ConvBnRelu(384, 384, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

        self.bulr0 = BlurPool(128, filt_size=3, stride=1)
        self.bulr1 = BlurPool(112, filt_size=3, stride=1)
        self.bulr2 = BlurPool(256, filt_size=2, stride=1)
        self.bulr3 = BlurPool(384, filt_size=1, stride=1)

        # self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        x = self.model(x)
        print(x.shape)
        x = x + self.bulr0(x)

        x = self.osa_0( x )
        res.append(x)

        x = x + self.bulr1(x)
        x = self.osa_1(x)
        res.append(x)

        x = x + self.bulr2(x)
        x = self.osa_2(x)
        res.append(x)

        x = x + self.bulr3(x)
        x = self.osa_3(x)
        res.append(x)
        return res

    def forward_pyramids(self, x: torch.Tensor) -> collections.OrderedDict:
        """
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.
        Usage:
        >>> net = VoVNet("vovnet-19-slim-dw", num_classes=1000)
        >>> net.delete_classification_head()
        >>> with torch.no_grad():
        ...    out = net.forward_pyramids(torch.randn(1, 3, 512, 512))
        >>> [level.shape[-1] for level in out.values()]  # Check the height/widths of levels
        [256, 128, 64, 32, 16]
        >>> [level.shape[1] for level in out.values()]  == net._out_feature_channels
        True
        """
        levels = collections.OrderedDict()
        levels[1] = self.model.stem(x)
        levels[2] = self.model.OSA_2(levels[1])
        levels[3] = self.model.OSA_3(levels[2])
        levels[4] = self.model.OSA_4(levels[3])
        levels[5] = self.model.OSA_5(levels[4])
        return levels

    def delete_classification_head(self) -> None:
        """ Call this before using model as an object detection backbone. """
        del self.model.classifier

    def get_pyramid_channels(self) -> None:
        """ Return the number of channels for each pyramid level. """
        return self._out_feature_channels
