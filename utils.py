import torch


def iou_width_height(boxes1, boxes2):
    """
    Calculate the IoU of two sets of boxes that are in the center-size notation.
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    return intersection / union