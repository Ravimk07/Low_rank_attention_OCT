import torch
import random
import numpy as np
import os
import torch.nn as nn
import glob
import tifffile as tiff
import torch.nn.functional as F


from torch.autograd import Variable
import scipy.misc
import tensorflow as tf
import errno
import imageio
import torchvision.transforms.functional as transform

from copy import deepcopy
from torch import autograd
from torch.autograd import Variable
from NNMetrics import segmentation_scores, f1_score, hd95, preprocessing_accuracy, intersectionAndUnion
from PIL import Image
from torch.utils import data


def test(data, model, device, class_no, save_location):
    # data: data loader of test data
    # model: loaded model
    # device: cpu or gpu
    # class_no: classes number
    # save_location: for saving tested results

    model.eval()

    data_testoutputs = []

    with torch.no_grad():
        #
        f1 = 0
        test_iou = 0
        # test_h_dist_1 = 0
        recall = 0
        precision = 0
        mse = 0
        # ==============================================
        evaluate_index_all = range(0, len(data) - 1)
        #
        for j, (testimg, testlabel, testimgname) in enumerate(data):
            #
            # ========================================================================
            # ========================================================================
            testimg = torch.from_numpy(testimg).to(device=device, dtype=torch.float32)
            testlabel = torch.from_numpy(testlabel).to(device=device, dtype=torch.float32)
            #
            c, h, w = testimg.size()
            testimg = testimg.expand(1, c, h, w)
            #
            testoutput_original = model(testimg)
            #
            if class_no == 2:
                #
                testoutput = torch.sigmoid(testoutput_original.view(1, h, w))
                testoutput = (testoutput > 0.5).float()
                data_testoutputs.append(testoutput)
                #
            else:
                #
                _, testoutput = torch.max(testoutput_original, dim=1)
                data_testoutputs.append(testoutput)

            # mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mean_iu_ = intersectionAndUnion(testoutput.cpu().detach(), testlabel.cpu().detach(), class_no)

            f1_, recall_, precision_ = f1_score(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mse_ = (np.square(testlabel.cpu().detach().numpy() - testoutput.cpu().detach().numpy())).mean()

            f1 += f1_
            test_iou += mean_iu_
            recall += recall_
            precision += precision_
            mse += mse_
            #
            #
            # # Plotting segmentation:
            # testoutput_original = np.asarray(testoutput_original.cpu().detach().numpy(), dtype=np.uint8)
            # testoutput_original = np.squeeze(testoutput_original, axis=0)
            # testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)
            # #
            # if class_no == 2:
            #     segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
            #     #
            #     segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
            #     segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #     segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #     #
            # else:
            #     segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
            #     if class_no == 4:
            #         # multi class for brats 2018
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
            #         #
            #     elif class_no == 8:
            #         # multi class for cityscapes
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 255
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 153
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 51
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 255
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 255
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 102
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 178
            #         #
            #         segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
            #         segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 255
            #         segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
            #         #
            # prediction_name = 'seg_' + test_imagename + '.png'
            # full_error_map_name = os.path.join(prediction_map_path, prediction_name)
            # imageio.imsave(full_error_map_name, segmentation_map)
    #
    prediction_map_path = save_location + '/' + 'Results_map'
    #
    try:
        os.mkdir(prediction_map_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    # save numerical results:
    result_dictionary = {'Test IoU': str(test_iou / len(evaluate_index_all)),
                         'Test f1': str(f1 / len(evaluate_index_all)),
                         'Test recall': str(recall / len(evaluate_index_all)),
                         'Test Precision': str(precision / len(evaluate_index_all)),
                         'Test MSE': str(mse / len(evaluate_index_all))}

    ff_path = prediction_map_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    return test_iou / len(evaluate_index_all), \
           f1 / len(evaluate_index_all), \
           recall / len(evaluate_index_all), \
           precision / len(evaluate_index_all), \
           mse / len(evaluate_index_all), \
           data_testoutputs