import torch
import torch.nn as nn
import numpy as np
import os, sys
import glob
import tifffile as tiff
import torch.nn.functional as F
import scipy.misc
import tensorflow as tf
from PIL import Image
import math
from torch.autograd import Variable
from torchvision import transforms
from torch import Tensor, einsum
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
# ==========================================================================


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(recon_x, x, mu, logvar, input_resolution, KL_weight):
    #
    x = F.upsample(x, size=(input_resolution//2, input_resolution//2), mode='bilinear', align_corners=True)

    (b, c, h, w) = x.shape

    # MSE_loss = F.binary_cross_entropy(recon_x.view(-1), x.view(-1), reduction='sum')

    MSE_loss = nn.MSELoss(reduction='sum')(recon_x, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    KLD = KLD / b

    MSE_loss = MSE_loss / b

    KLD = KLD * KL_weight

    # KLD = torch.sqrt(KLD) + 1e-8

    # MSE_loss = torch.sqrt(MSE_loss) + 1e-8

    return MSE_loss + KLD, MSE_loss, KLD


def fbeta_loss(y_true, y_pred):
    # tune hyper-parameters here:
    # beta: ratio between precision and recall
    # threshold: above threshold are considered as positive
    # eps: smoothing term
    beta = 0.5
    threshold = 0.5
    eps = 0.1
    # ========
    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()
    # =====================
    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))
    fbeta_score = torch.mean((precision*recall).div(precision.mul(beta) + recall + eps).mul(1 + beta))

    return 1-fbeta_score

# ==========================


def dice_loss(input, target):
    smooth = 0.1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    # union = (torch.mul(iflat, iflat) + torch.mul(tflat, tflat)).sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
