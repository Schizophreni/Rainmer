"""
This file defines required loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmarks.SSIM import SSIM

ssim_metric = SSIM()

class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

def ssim_fidelity(restored, gt):
    return 1 - ssim_metric(restored, gt)

def pixel_fidelity(restored, gt):
    return F.l1_loss(restored, gt, size_average=True)

def multiscale_location_loss(predict_locations, target_locations):
    # predict locations: list of tensor with shape [B, C, H, W]
    # target locations: list of tensor with shape [B, H, W] with int64 numbers
    loss = 0.0
    for pred_loc, target_loc in zip(predict_locations, target_locations):
        # loss += F.cross_entropy(pred_loc, target_loc.long(), reduction="mean")
        loss += F.mse_loss(pred_loc, target_loc)
    return loss / len(predict_locations)

def contrastive_loss_cos(logits, weight=None):
    # contra loss for detail, illu, and degradation atoms
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
    batch_loss = F.cross_entropy(logits, labels, reduction="none")
    if weight is not None:
        batch_loss = batch_loss * weight
        return batch_loss.sum() / (1e-5 + weight.sum())
    return batch_loss.mean()

def contrastive_loss_l1(logits, weight):
    # contra loss for detail, illu, and degradation using L1 distance
    l = logits[:, 0] / (logits[:, 1:].sum(dim=[-1]) + 1e-7)
    return l.mean()