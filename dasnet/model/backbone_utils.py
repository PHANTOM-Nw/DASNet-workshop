import warnings
from typing import Callable, Dict, List, Optional, Union

from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.models import mobilenet, resnet
from torchvision.models._api import _get_enum_from_fn, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, IntermediateLayerGetter

import torch.nn.functional as F
from collections import OrderedDict


class BackboneWithCustomFPN(nn.Module):
    """
    Adds a selectable FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    During the process, the max scale of layer is constrainted with the target layer.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        target_layer (str): define which layer is the max layer scale
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        target_layer: str = 'P4',
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = CustomFeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            target_layer=target_layer,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


def _resnet_customfpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    target_layer: str = 'P4',
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:
    """
    Modify the ResNet backbone to integrate with FPN and add extra downsampling.
    """

    def adjust_stride_and_downsample(layer, block_idx, in_channels, out_channels):
        layer[block_idx].conv2.stride = (2, 2)
        if layer[block_idx].downsample is None:
            layer[block_idx].downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            layer[block_idx].downsample[0].stride = (2, 2)

    # Modify conv2_x (layer1) first block and conv3_x, conv4_x, conv5_x second blocks
    adjust_stride_and_downsample(backbone.layer1, 0, 256, 256)   # conv2_x
    # adjust_stride_and_downsample(backbone.layer2, 1, 512, 512)   # conv3_x
    # adjust_stride_and_downsample(backbone.layer3, 1, 1024, 1024) # conv4_x
    # adjust_stride_and_downsample(backbone.layer4, 1, 2048, 2048) # conv5_x

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")

    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 128

    if target_layer == 'P2':
        return BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        )
    else:
        return BackboneWithCustomFPN(
            backbone, return_layers, in_channels_list, out_channels, target_layer, extra_blocks=extra_blocks, norm_layer=norm_layer
        )


def _resnet_fpn_extractor_shallow(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithFPN:

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 128
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )


class CustomFeaturePyramidNetwork(FeaturePyramidNetwork):
    def __init__(self, in_channels_list, out_channels, target_layer="P4", extra_blocks=None, norm_layer=None):
        super().__init__(in_channels_list, out_channels, extra_blocks, norm_layer)
        self.target_layer = target_layer  # target layer

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        use target layer to 
        """
        names = list(x.keys())
        x = list(x.values())

        # top-down pathway
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

            # stop at target layer
            if names[idx] == self.target_layer:
                break

        # down-top pathway
        for idx in range(idx - 1, -1, -1):
            downsampled = F.max_pool2d(self.get_result_from_inner_blocks(x[idx], idx), kernel_size=2, stride=2)
            last_inner = results[0] + downsampled
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # extra layer
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        return OrderedDict([(k, v) for k, v in zip(names, results)])