# utils.py
# This file contains utility functions used throughout the project, including model parameter counting and ROI pooling helpers.

import torch
import torch.nn.functional as F
import torch.autograd as ag

def get_n_params(model):
    """
    Count the total number of parameters in a PyTorch model.
    Args:
        model: PyTorch model
    Returns:
        int: Total number of parameters
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def adaptive_max_pool(input, size):
    """
    Wrapper for PyTorch's adaptive max pooling.
    Args:
        input: Input tensor
        size: Output size (tuple)
    Returns:
        Tensor: Pooled output
    """
    return F.adaptive_max_pool2d(input, size)

def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    """
    Perform ROI pooling on a batch of feature maps and ROIs.
    Args:
        input: Feature map tensor
        rois: Regions of interest (N x 5 tensor)
        size: Output size (tuple)
        spatial_scale: Scale factor for ROI coordinates
    Returns:
        Tensor: ROI pooled output
    """
    assert (rois.dim() == 2)
    assert (rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        output.append(adaptive_max_pool(im, size))
    return torch.cat(output, 0)

def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    """
    Perform ROI pooling for one ROI per image in a batch.
    Args:
        input: Feature map tensor
        rois: Regions of interest (N x 4 tensor)
        size: Output size (tuple)
        spatial_scale: Scale factor for ROI coordinates
    Returns:
        Tensor: ROI pooled output
    """
    assert (rois.dim() == 2)
    assert len(input) == len(rois)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
        output.append(adaptive_max_pool(im, size))
    return torch.cat(output, 0) 