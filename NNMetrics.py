import torch
import torch.nn as nn
import numpy as np
import scipy.spatial
import os, sys
import tensorflow as tf
from PIL import Image
import math
from torch.autograd import Variable
from torch.utils import data
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import _ni_support

from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy import ndimage
# good references:
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
#-----------------------------------------------
# ==============================================================================================================
# accuracy:


def preprocessing_accuracy(label_true, label_pred, n_class):
    # thresholding predictions:
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype='int8')
    label_true = np.asarray(label_true, dtype='int8')

    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 7)

    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)

    return label_pred, label_true

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py


def _fast_hist(label_true, label_pred, n_class):

    label_pred, label_true = preprocessing_accuracy(label_true, label_pred, n_class)

    hist = np.bincount(n_class * label_true + label_pred, minlength=n_class**2).reshape(n_class, n_class)

    return hist


def segmentation_scores(label_trues, label_preds, n_class):

    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return mean_iu
# ==================================================================================


def f1_score(label_gt, label_pred, n_class):
    # threhold = torch.Tensor([0])

    label_pred, label_gt = preprocessing_accuracy(label_gt, label_pred, n_class)
    #
    assert len(label_gt) == len(label_pred)

    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)

    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()

    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))
    f1_metric = 2 * (recall * precision) / (recall + precision + 1e-8)
    #
    return f1_metric.mean(), recall.mean(), precision.mean()


# ========================================================================
# reference :http://loli.github.io/medpy/_modules/medpy/metric/binary.html


def __surface_distances(result, reference, no_class, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result, reference = preprocessing_accuracy(reference, result, no_class)
    #
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, n_class, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, n_class, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, n_class, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95