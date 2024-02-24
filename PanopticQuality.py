import torch
from monai.metrics.utils import do_metric_reduction, remap_instance_id
from monai.utils import MetricReduction, ensure_tuple, optional_import
linear_sum_assignment, _ = optional_import("scipy.optimize", name="linear_sum_assignment")

# Panoptic Quality

def compute_panoptic_quality(
    pred,
    gt,
    metric_name= "pq",
    remap = True,
    match_iou_threshold= 0.5,
    smooth_numerator = 1e-6,
    output_confusion_matrix= False):

    if gt.shape != pred.shape:
        raise ValueError(f"pred and gt should have same shapes, got {pred.shape} and {gt.shape}.")
    if match_iou_threshold <= 0.0 or match_iou_threshold > 1.0:
        raise ValueError(f"'match_iou_threshold' should be within (0, 1], got: {match_iou_threshold}.")
    
    gt = gt.int()
    pred = pred.int()

    if remap is True:
        gt = remap_instance_id(gt)
        pred = remap_instance_id(pred)

    pairwise_iou, true_id_list, pred_id_list = _get_pairwise_iou(pred, gt, device=pred.device)
    paired_iou, paired_true, paired_pred = _get_paired_iou(
        pairwise_iou, match_iou_threshold, device=pairwise_iou.device
    )

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp, fp, fn = len(paired_true), len(unpaired_pred), len(unpaired_true)
    iou_sum = paired_iou.sum()

    if output_confusion_matrix:
        return torch.as_tensor([tp, fp, fn, iou_sum], device=pred.device)

    metric_name = _check_panoptic_metric_name(metric_name)
    if metric_name == "rq":
        return torch.as_tensor(tp / (tp + 0.5 * fp + 0.5 * fn + smooth_numerator), device=pred.device)
    if metric_name == "sq":
        return torch.as_tensor(iou_sum / (tp + smooth_numerator), device=pred.device)
    return torch.as_tensor(iou_sum / (tp + 0.5 * fp + 0.5 * fn + smooth_numerator), device=pred.device)


def _get_id_list(gt):
    id_list = list(gt.unique())
    # ensure id 0 is included
    if 0 not in id_list:
        id_list.insert(0, torch.tensor(0).int())

    return id_list

def _get_pairwise_iou(pred, gt, device= "cpu"):

    pred_id_list = _get_id_list(pred)
    true_id_list = _get_id_list(gt)

    pairwise_iou = torch.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=torch.float, device=device)
    true_masks = []
    pred_masks = []

    for t in true_id_list[1:]:
        t_mask = torch.as_tensor(gt == t, device=device).int()
        true_masks.append(t_mask)

    for p in pred_id_list[1:]:
        p_mask = torch.as_tensor(pred == p, device=device).int()
        pred_masks.append(p_mask)

    for true_id in range(1, len(true_id_list)):
        t_mask = true_masks[true_id - 1]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(pred_true_overlap.unique())
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id - 1]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    return pairwise_iou, true_id_list, pred_id_list


def _get_paired_iou(
    pairwise_iou, match_iou_threshold= 0.5, device = "cpu"):

    if match_iou_threshold >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou_threshold] = 0.0
        paired_true, paired_pred = torch.nonzero(pairwise_iou)[:, 0], torch.nonzero(pairwise_iou)[:, 1]
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1

        return paired_iou, paired_true, paired_pred

    pairwise_iou = pairwise_iou.cpu().numpy()
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    paired_iou = pairwise_iou[paired_true, paired_pred]
    paired_true = torch.as_tensor(list(paired_true[paired_iou > match_iou_threshold] + 1), device=device)
    paired_pred = torch.as_tensor(list(paired_pred[paired_iou > match_iou_threshold] + 1), device=device)
    paired_iou = paired_iou[paired_iou > match_iou_threshold]

    return paired_iou, paired_true, paired_pred

def _check_panoptic_metric_name(metric_name):
     metric_name = metric_name.replace(" ", "_")
     metric_name = metric_name.lower()
     if metric_name in ["panoptic_quality", "pq"]:
         return "pq"
     if metric_name in ["segmentation_quality", "sq"]:
         return "sq"
     if metric_name in ["recognition_quality", "rq"]:
         return "rq"
     raise ValueError(f"metric name: {metric_name} is wrong, please use 'pq', 'sq' or 'rq'.")
