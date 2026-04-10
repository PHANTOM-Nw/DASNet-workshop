from .maskrcnn import MaskRCNN
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn
from .backbone_utils import _resnet_customfpn_extractor
from typing import Tuple

def build_model(
    model_name: str = "maskrcnn_resnet50_selectable_fpn",
    num_classes: int = None,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    target_layer: str = 'P4',
    box_roi_pool_size: int = 7,
    mask_roi_pool_size: Tuple[int, int] = (42, 42),
    **kwargs
) -> MaskRCNN:

    # MaskRCNN-ResNet50-FPN pretrained weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None

    # ResNet pretrained weights
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    # backbone
    if model_name == "maskrcnn_resnet50_selectable_fpn":
        backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=norm_layer)
        backbone = _resnet_customfpn_extractor(backbone, trainable_backbone_layers, target_layer)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['1', '2', '3', '4'],
        output_size=box_roi_pool_size,
        sampling_ratio=2,
    )
    model.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['1', '2', '3', '4'],
        output_size=mask_roi_pool_size,
        sampling_ratio=2,
    )

    if model_name == "maskrcnn_resnet50_selectable_fpn":
        if target_layer == 'P2':
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        elif target_layer == 'P3':
            anchor_sizes = ((64,), (64,), (128,), (256,), (512,))
        elif target_layer == 'P4':
            anchor_sizes = ((128,), (128,), (128,), (256,), (512,))
        elif target_layer == 'P5':
            anchor_sizes = ((256,), (256,), (256,), (256,), (512,))
        else:
            anchor_sizes = ((512,), (512,), (512,), (512,), (512,))
        # Czech DAS events are extremely time-long / channel-short:
        # bbox H/W distribution on the train set has p5=0.003, p95=0.188,
        # with 95% of GT boxes outside the stock (0.5, 1, 2) range.
        # Replace with 5 log-quantile ratios covering p5..p95 so the RPN
        # actually has anchors that can match Czech bbox shapes.
        aspect_ratios = ((0.003, 0.008, 0.024, 0.067, 0.188),) * len(anchor_sizes)
        model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    return model

