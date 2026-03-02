import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext
import csv
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter
import utils
import json
from dasnet.data.continuous_dataset import DASNet__ContinuousDataset
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from dasnet.model.dasnet import build_model
import fsspec
import torch.distributed as dist
from torchvision.ops import box_iou
import time

label_map = {
    1: "Blue whale A",
    2: "Blue whale B",
    3: "Fin whale",
    4: "Others",
    5: "Blue whale D",
    6: "T wave",
    7: "Ship",
    8: "P wave",
    9: "S wave",
}
logger = logging.getLogger()

def get_files(data_list):
    if data_list == '':
        if utils.is_main_process():
            print("Searching files ...")
        # GCS Key path
        default_key_path = "./test_skypilot/x-berkeley-mbari-das-8c2333fca1b2.json"
        fs = fsspec.filesystem("gcs", token=default_key_path)

        # hdf5 file name
        hdf5_files = []
        folders = fs.ls("berkeley-mbari-das/")
        for folder in folders:
            if folder.split("/")[-1] in ["ContextData", "MBARI_cable_geom_dx10m.csv"]:
                continue
            years = fs.ls(folder)
            for year in years:
                jdays = fs.ls(year)
                for jday in jdays:
                    files = fs.ls(jday)
                    for file in files:
                        if file.endswith(".h5") and os.path.basename(file).startswith("MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2023-11-0"): #MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2024-12-3
                            hdf5_files.append(file)
        if utils.is_main_process():
            print(f'Total file number: {len(hdf5_files)}')
    else:
        hdf5_files = pd.read_csv(args.data_list)
        if utils.is_main_process():
            print(f'Total file number: {len(hdf5_files)}')

    return hdf5_files
        

def extract_peak_points(matrix, x_range, y_range, threshold=0.5, sigma=5, is_filt=True, is_gauss=True):
    if is_gauss:
        matrix = gaussian_filter(matrix, sigma=sigma)

    points = []
    points_value = []
    for y in range(matrix.shape[0]):
        row = matrix[y]
        peak_x = np.argmax(row)
        peak_value = row[peak_x]
        if peak_value > threshold and x_range[0] < peak_x < x_range[1] and y_range[0] < y < y_range[1]:
            points.append([float(peak_x), float(y)])
            points_value.append(float(peak_value))

    if is_filt:
        lines = defaultdict(list)
        for x, y in points:
            lines[x].append(y)

        filtered_points = []
        for x, y_list in lines.items():
            y_list.sort()
            mid_index = len(y_list) // 2
            sampled_y = y_list[mid_index]
            filtered_points.append([x, sampled_y])
    else:
        filtered_points = sorted(points, key=lambda point: point[1])

    if len(filtered_points) == 0:
        filtered_points = []
    
    return filtered_points, points_value


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two binary masks (shape: [H, W])
    """
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return intersection / union if union > 0 else torch.tensor(0.0, device=mask1.device)


def mask_agnostic_nms_single_image(
    prediction: dict,
    box_iou_thresh: float = 0.8,
    mask_iou_thresh: float = 0.8,
    mask_threshold: float = 0.5
) -> dict:
    """
    Applies class-agnostic, mask-aware NMS to a single image's prediction.
    Args:
        prediction: Dict with keys 'boxes', 'scores', 'labels', 'masks'
        box_iou_thresh: IoU threshold for box overlap
        mask_iou_thresh: IoU threshold for mask overlap
        mask_threshold: Threshold for converting soft masks to binary masks
    Returns:
        Filtered prediction dict with same format
    """
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    masks = prediction["masks"].squeeze(1)  # [N, H, W]

    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())
        if idxs.numel() == 1:
            break

        current_box = boxes[current].unsqueeze(0)  # [1, 4]
        rest_boxes = boxes[idxs[1:]]                # [N-1, 4]
        box_ious = box_iou(current_box, rest_boxes).squeeze(0)  # [N-1]

        current_mask = (masks[current] >= mask_threshold)       # [H, W]
        rest_masks = masks[idxs[1:]] >= mask_threshold          # [N-1, H, W]
        mask_ious = torch.tensor([
            compute_mask_iou(current_mask, m) for m in rest_masks
        ], device=masks.device)

        remove_mask = (box_ious > box_iou_thresh) | (mask_ious > mask_iou_thresh)
        idxs = idxs[1:][~remove_mask]

    # Return filtered results
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "masks": masks[keep].unsqueeze(1)  # restore [N, 1, H, W]
    }


def postprocess_dasnet(filenames, output, alpha=(1000 / 200) / 5.1):
    """Post-process the DASNet model output, handling mask cropping."""
    batch_size = len(filenames)

    results = []

    # second nms to remove different categories overlapped instances
    output = [mask_agnostic_nms_single_image(out) for out in output]

    for i in range(batch_size):
        result = {
            "boxes": output[i]["boxes"].detach().cpu().numpy(),
            "scores": output[i]["scores"].detach().cpu().numpy(),
            "labels": output[i]["labels"].detach().cpu().numpy(),
        }

        # orignal data shape
        masks = output[i]['masks']
        H_img, W_img = masks.shape[-2:]

        # handling mask
        boxes = output[i]["boxes"]
        masks = output[i]["masks"]  # (N, 1, H_crop, W_crop)

        final_masks = torch.zeros((len(masks), H_img, W_img), device=masks.device)

        for j, (box, mask) in enumerate(zip(boxes, masks)):
            x_min, y_min, x_max, y_max = box.int()
            cropped_mask = mask[:, y_min:y_max, x_min:x_max]  # Crop mask to the box region

            height = y_max - y_min
            cropped_width = int(height * alpha)

            # resize mask
            resized_mask = F.interpolate(
                cropped_mask.unsqueeze(0),
                size=(height, cropped_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            full_box_mask = torch.zeros((height, x_max - x_min), device=masks.device)

            # ensure resized_mask dimensions match full_box_mask
            cropped_width = min(cropped_width, full_box_mask.shape[1])
            resized_mask = resized_mask[:, :, :cropped_width]

            full_box_mask[:, :cropped_width] = resized_mask
            final_masks[j, y_min:y_max, x_min:x_max] = full_box_mask

        result["masks"] = final_masks.cpu().numpy()
        results.append(result)

    return filenames, results


def save_to_labelme_format(file_name, selected_results, peak_points_list, peak_scores_list, output_dir):
    shapes = []

    for i in range(len(selected_results["scores"])):
        box = selected_results["boxes"][i]
        score = float(selected_results["scores"][i])
        label_id = int(selected_results["labels"][i])
        label_name = label_map.get(label_id, str(label_id))
        peak_points = peak_points_list[i]
        peak_scores = peak_scores_list[i]

        # Box
        box_shape = {
            "label": "box",
            "points": [
                [float(box[0]), float(box[1])],
                [float(box[2]), float(box[3])]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
            "score": score
        }
        shapes.append(box_shape)

        # Peak points (LineStrip)
        if peak_points == []:
            peak_points = [[0, 0], [1, 1]]
            peak_scores = [0, 0]
        if label_id in [1, 5, 6, 7]:
            peak_points = [[0, 0], [1, 1]]
            peak_scores = [0, 0]
        line_shape = {
            "label": label_name,
            "points": [[float(x), float(y)] for x, y in peak_points],
            "point_scores": [x for x in peak_scores],
            "group_id": None,
            "description": "",
            "shape_type": "linestrip",
            "flags": {},
            "mask": None,
            "score": score
        }
        shapes.append(line_shape)

        # Confident areas (rectangle list)
        xmin = float(box[0])
        xmax = float(box[2])

        maskpoint = sorted(peak_points, key=lambda p: p[1])
        current_group = []
        confident_areas = []

        for j, (x, y) in enumerate(maskpoint):
            if not current_group or y <= current_group[-1][1] + 15:
                current_group.append((x, y))
            else:
                ymin = current_group[0][1]
                ymax = current_group[-1][1]
                if ymin == ymax:
                    if ymax < float(box[3]):
                        ymax += 1
                    else:
                        ymin -= 1
                if ymax - ymin > 50:
                    confident_areas.append((xmin, xmax, ymin, ymax))
                current_group = [(x, y)]

        if current_group:
            ymin = current_group[0][1]
            ymax = current_group[-1][1]
            if ymax - ymin > 50:
                confident_areas.append((xmin, xmax, ymin, ymax))

        for (xmin, xmax, ymin, ymax) in confident_areas:
            ca_shape = {
                "label": "confident_area",
                "points": [
                    [xmin, ymin],
                    [xmax, ymax]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
                "score": score
            }
            shapes.append(ca_shape)

    result = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name + ".png",
        "imageData": "",
        "imageHeight": 2845,
        "imageWidth": 12000
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, file_name + ".json"), "w") as f:
        json.dump(result, f, indent=2)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
def plot_das_predictions(input_data, predictions, save_path, score_threshold=0.8, mask_threshold=0.25):
    fig, axes = plt.subplots(3, 1, figsize=(7*7/2*0.6, 3 * 7*7/8*0.6))

    for idx, ax in enumerate(axes):
        image_data = np.array(input_data[idx], dtype=np.float32)
        # print(np.percentile(image_data, 90))
        plt.sca(ax)
        plt.imshow(np.flipud(image_data), cmap='seismic', vmin=np.percentile(image_data, 10), vmax=np.percentile(image_data, 90), rasterized=True)

        boxes = predictions['boxes']
        masks = predictions['masks']
        labels = predictions['labels']
        scores = predictions['scores']

        H = image_data.shape[0]

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score > score_threshold:
                x1, y1, x2, y2 = box

                y1_flipped = H - y2
                y2_flipped = H - y1
                x1_flipped, x2_flipped = x1, x2
                x1, y1, x2, y2 = x1_flipped, y1_flipped, x2_flipped, y2_flipped

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=4, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)

                label_text = label_map[label] if label in label_map else f'class {label}'
                plt.text(x1 + 40, y2 - 150, f'{label_text}\n{score:.2f}', fontsize=10)

                mask = mask.squeeze()

                smooth_mask = mask = np.flipud(mask)

                alpha_mask = smooth_mask.copy()

                if label in [2, 3, 8, 9]: # 4
                    alpha_mask[alpha_mask <= mask_threshold] = 0
                    alpha_mask[alpha_mask > mask_threshold] = 0.5
                else:
                    alpha_mask[:] = 0

                jet_colormap = plt.get_cmap('jet')
                rgba_mask = jet_colormap(smooth_mask)
                rgba_mask[..., 3] = alpha_mask

                ax.imshow(rgba_mask)

                peak_points, _ = extract_peak_points(
                    mask, [x1, x2], [y1, y2],
                    threshold=0.5, is_filt=False, is_gauss=True
                )

                if len(peak_points) > 0 and label in [2, 3, 8, 9]: #4
                    peak_x, peak_y = zip(*peak_points)
                    plt.scatter(peak_x, peak_y, color='white', s=1)

        plt.xlim(0, 6000)
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000],
                   ["0", "10", "20", "30", "40", "50", "60"])

        plt.ylim(0, 1422)
        plt.yticks(
            [1422-1300, 1422-1050, 1422-800, 1422-550, 1422-300, 1422-50],
            ["10000", "9500", "9000", "8500", "8000", "7500"]
        )
        plt.ylabel("Channel number")

        if idx == 2:
            plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def pred_dasnet(args, model, data_loader, pick_path, figure_path):
    """Run inference on DASNet dataset using Mask R-CNN."""
    
    model.eval()
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)

    with torch.inference_mode():
        for images, filenames in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            a = time.time()
            images = [img.to(args.device) for img in images]
            
            with ctx:
                output = model(images)
                # print(f'inference time: {time.time()-a}')
                b = time.time()
                filenames, output = postprocess_dasnet(filenames, output)
                # print(f'postprocess time: {time.time()-b}')

            for i in range(len(filenames)):
                file_name = os.path.basename(filenames[i])
                scores = output[i]["scores"]
                
                # only save predictions with scores higher than threshold
                mask_threshold = args.min_prob
                keep_idx = scores > mask_threshold
                selected_results = {
                    "boxes": output[i]["boxes"][keep_idx],
                    "scores": scores[keep_idx],
                    "masks": output[i]["masks"][keep_idx],
                    "labels": output[i]["labels"][keep_idx],
                }

                # empty file for no predictions
                if len(selected_results["scores"]) == 0:
                    # with open(os.path.join(pick_path, file_name + ".csv"), "a"):
                    #     pass
                    continue

                # save peak points instead of masks
                peak_points_list = []
                peak_points_values = []
                for j in range(len(selected_results["scores"])):
                    mask = selected_results["masks"][j]  # (1, H, W) -> (H, W)
                    x_min, y_min, x_max, y_max = selected_results["boxes"][j]
                    x_range = (x_min, x_max)
                    y_range = (y_min, y_max)

                    peak_points, peak_point_values = extract_peak_points(mask, x_range, y_range, threshold=0.5, is_filt=False, is_gauss=True)
                    peak_points_list.append(peak_points)
                    peak_points_values.append(peak_point_values)

                with open(os.path.join(pick_path, file_name + ".csv"), mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["x_min", "y_min", "x_max", "y_max", "score", "label", "peak_points", "peak_scores"])
                    for j in range(len(selected_results["scores"])):
                        box = selected_results["boxes"][j]
                        score = selected_results["scores"][j]
                        label = selected_results["labels"][j]
                        peak_points = json.dumps(peak_points_list[j])
                        peak_scores = json.dumps(peak_points_values[j])

                        writer.writerow([box[0], box[1], box[2], box[3], score, label, peak_points, peak_scores])

                save_to_labelme_format(file_name, selected_results, peak_points_list, peak_points_values, pick_path)

                if args.plot_figure:
                    plot_das_predictions(
                        images[i].cpu().numpy(),
                        selected_results,
                        os.path.join(figure_path, file_name + ".pdf"),
                        args.min_prob
                    )

    if args.distributed:
        torch.distributed.barrier()
        dist.destroy_process_group()

    return 0


def main(args):
    result_path = args.result_path
    figure_path = os.path.join(result_path, f"figures_{args.model}")

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    device = torch.device(args.device)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # if args.model in ["dasnet"]:...
    dataset = DASNet__ContinuousDataset(
        get_files(args.data_list),
        key_path=args.key_path if hasattr(args, 'key_path') else None,
        target_size=(1422, 6000),
        pred_object=args.object,
        channel_range=[3000, 5845]#[7400, 10245]
    )
    sampler = torch.utils.data.DistributedSampler(dataset) if args.distributed else None

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, mp.cpu_count()),
        collate_fn=None,
        drop_last=False,
    )

    model = build_model( # eqnet.models.__dict__[args.model].build_model(
        model_name="maskrcnn_resnet50_selectable_fpn",
        num_classes=1+10,
        pretrained=False,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        target_layer='P4',
        box_roi_pool_size=7,
        mask_roi_pool_size=(42, 42),
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_positive_fraction=0.25,
        max_size=6000,
        min_size=1422
    )
    logger.info("Model:\n{}".format(model))

    model.to(device)

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        raise ("Missing pretrained model for this location")
        # print(f"Loading pretrained model from: {args.model_path}")
        # model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu])

    model.eval()
    pred_dasnet(args, model, data_loader, result_path, figure_path)

import argparse
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="DASNet Mask R-CNN Inference", add_help=add_help)

    # **Model-related parameters**
    parser.add_argument("--model", default="dasnet", type=str, help="Model name")
    # parser.add_argument("--model_path", type=str, default="", help="Path to the pre-trained model (.pth file)")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resuming inference")

    # **Device & computation**
    parser.add_argument("--device", default="cuda", type=str, help="Device to use: cuda / cpu")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU")
    parser.add_argument("--workers", default=4, type=int, help="Number of data loading workers")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (Automatic Mixed Precision) inference")

    # **Distributed inference**
    parser.add_argument("--distributed", action="store_true", help="Enable distributed inference")
    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="URL for setting up distributed inference")

    # **Data input**
    parser.add_argument("--data_list", type=str, default="", help="CSV file with the path of the input data")
    parser.add_argument("--object", type=str, default="default", help="Specify target files for inference (default: all)")
    parser.add_argument("--key_path", type=str, default=None, help="Path to the GCS access key (if using Google Cloud data)")

    # **Prediction settings**
    parser.add_argument("--result_path", type=str, default="results", help="Path to save inference results")
    parser.add_argument("--min_prob", default=0.5, type=float, help="Confidence threshold (predictions below this value will be discarded)")
    parser.add_argument("--plot_figure", action="store_true", help="Save prediction visualization images")

    # **DAS-specific settings**
    # parser.add_argument("--nt", default=4000, type=int, help="Number of time samples")
    # parser.add_argument("--nx", default=948, type=int, help="Number of spatial samples")
    # parser.add_argument("--cut_patch", action="store_true", help="Enable patching for continuous data")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing files that already have results")

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)