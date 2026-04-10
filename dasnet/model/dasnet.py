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
        # Czech DAS events are time-long / channel-short.  After the axis
        # swap in das.py (COCO x=time,y=ch → model x=ch,y=time), model-space
        # boxes are TALL (y=time) and NARROW (x=channel).  Torchvision anchors
        # use aspect_ratio = H/W, so we need ratios >> 1.
        # Derived from the train-set bbox distribution (model-space H/W):
        #   p5=5.3  p25=14.5  p50=64  p75=152  p95=333
        aspect_ratios = ((5.3, 14.9, 41.7, 125.0, 333.0),) * len(anchor_sizes)
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        model.rpn.anchor_generator = anchor_gen
        # RPNHead conv layers must match the new anchor count per location
        from torchvision.models.detection.rpn import RPNHead
        num_anchors = anchor_gen.num_anchors_per_location()[0]
        out_channels = model.backbone.out_channels
        model.rpn.head = RPNHead(out_channels, num_anchors)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    return model

