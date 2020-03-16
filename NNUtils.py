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
# ================================================================================================
# all ramp functions are from:
# https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/ramps.py


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def dynamic_ema(model1, model2, loss1, loss2, ratio, mode):
    #
    if mode == 'dynamic':
        maximum_acc = min(loss1, loss2)
        # copy the weights of the best model to others
        if maximum_acc == loss1:
            #
            for conv_layer_1, conv_layer_2 in zip(model1.modules(), model2.modules()):
                if isinstance(conv_layer_1, nn.Conv2d) and isinstance(conv_layer_2, nn.Conv2d):
                    conv_layer_2.weight.data = (1 - ratio) * conv_layer_1.weight.data.detach() + ratio * conv_layer_2.weight.data
                    if conv_layer_1.bias is True and conv_layer_2.bias is True:
                        conv_layer_2.bias.data = (1 - ratio) * conv_layer_1.bias.data.detach() + ratio * conv_layer_2.bias.data

        elif maximum_acc == loss2:
            #
            for conv_layer_1, conv_layer_2 in zip(model1.modules(), model2.modules()):
                if isinstance(conv_layer_1, nn.Conv2d) and isinstance(conv_layer_2, nn.Conv2d):
                    conv_layer_1.weight.data = (1 - ratio) * conv_layer_2.weight.data.detach() + ratio * conv_layer_1.weight.data
                    if conv_layer_1.bias is True and conv_layer_2.bias is True:
                        conv_layer_1.bias.data = (1 - ratio) * conv_layer_2.bias.data.detach() + ratio * conv_layer_1.bias.data

    elif mode == 'static':
        # model1 is always the teacher
        # ratio_static = 0.99
        for conv_layer_1, conv_layer_2 in zip(model1.modules(), model2.modules()):
            if isinstance(conv_layer_1, nn.Conv2d) and isinstance(conv_layer_2, nn.Conv2d):
                conv_layer_2.weight.data = (1 - ratio) * conv_layer_1.weight.data.detach() + ratio * conv_layer_2.weight.data
                if conv_layer_1.bias is True and conv_layer_2.bias is True:
                    conv_layer_2.bias.data = (1 - ratio) * conv_layer_1.bias.data.detach() + ratio * conv_layer_2.bias.data

    elif mode == 'average':

        for conv_layer_1, conv_layer_2 in zip(model1.modules(), model2.modules()):
            if isinstance(conv_layer_1, nn.Conv2d) and isinstance(conv_layer_2, nn.Conv2d):
                conv_layer_1.weight.data = (conv_layer_1.weight.data + conv_layer_2.weight.data.detach()) / 2
                conv_layer_2.weight.data = conv_layer_1.weight.data.detach()
                if conv_layer_1.bias is True and conv_layer_2.bias is True:
                    conv_layer_1.bias.data = (conv_layer_1.bias.data + conv_layer_2.bias.data.detach()) / 2
                    conv_layer_2.bias.data = conv_layer_1.bias.data.detach()


def create_model(model_type, device, student_mode=True):
    model = model_type
    model.to(device)
    if student_mode is True:
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                if layer.bias is True:
                    layer.bias.requires_grad = False
    return model


def getData_OCT(data_directory, train_batchsize, shuffle_mode, augmentation_train, augmentation_test):

    train_image_folder = data_directory + 'train/images'
    train_label_folder = data_directory + 'train/masks'

    validate_image_folder = data_directory + 'val/images'
    validate_label_folder = data_directory + 'val/masks'

    test_image_folder_1 = data_directory + 'test_1/images'
    test_label_folder_1 = data_directory + 'test_1/masks'

    test_image_folder_2 = data_directory + 'test_2/images'
    test_label_folder_2 = data_directory + 'test_2/masks'

    train_dataset = CustomDataset_OCT(train_image_folder, train_label_folder, teacher_student=False, transforms=augmentation_train)
    validate_dataset = CustomDataset_OCT(validate_image_folder, validate_label_folder, teacher_student=False, transforms=augmentation_test)
    test_dataset_1 = CustomDataset_OCT(test_image_folder_1, test_label_folder_1, teacher_student=False, transforms=augmentation_test)
    test_dataset_2 = CustomDataset_OCT(test_image_folder_2, test_label_folder_2, teacher_student=False, transforms=augmentation_test)

    num_cores = 4

    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=shuffle_mode, num_workers=2*num_cores, drop_last=False)
    valloader = data.DataLoader(validate_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False)

    return trainloader, train_dataset, valloader, test_dataset_1, test_dataset_2


class CustomDataset_OCT(torch.utils.data.Dataset):

    def __init__(self, imgs_folder, labels_folder, teacher_student, transforms):

        # 1. Initialize file paths or a list of file names.
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.transform = transforms
        self.teacher_student = teacher_student

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        all_images = glob.glob(os.path.join(self.imgs_folder, '*.jpg'))
        all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy'))
        # sort all in the same order, very important
        all_labels.sort()
        all_images.sort()

        image = imageio.imread(all_images[index])
        image = np.array(image, dtype='float32')
        label = np.load(all_labels[index])
        label = np.array(label, dtype='float32')
        #
        image_dim_total = len(image.shape)
        #
        if image_dim_total == 2:
            #
            (height, width) = image.shape
            #
        elif image_dim_total == 3:
            #
            (height, width, c) = image.shape
            image = image[:, :, 0]
            #
        label = label.reshape(1, height, width)
        image = image.reshape(1, height, width)
        #
        # if label.max() > 1.0:
        #     #
        #     label = label + 1.0
        #
        # get the name of the file:
        labelname = all_images[index]
        labelname, extenstion = os.path.splitext(labelname)
        dirpath_parts = labelname.split('/')
        labelname = dirpath_parts[-1]
        # Output two perturbations of the same input
        # Augmentation:
        if self.teacher_student is True:
            # two data inputs for teacher and student
            # one with flipping, another one without flipping
            image_augmented = np.copy(image)
            label_augmented = np.copy(label)
            #
            # for channel in range(1):
            # image_augmented[channel, :, :] = np.flip(image_augmented[channel, :, :], axis=1).copy()
            image_augmented = np.flip(image_augmented, axis=2).copy()
            label_augmented = np.flip(label_augmented, axis=2).copy()
            # =======================================================
            output_augmentation = random.random()

            if output_augmentation > 0.5:
                return image, label, image_augmented, label_augmented, labelname
            else:
                return image_augmented, label_augmented, image, label, labelname

        else:
            # only single output for one model
            if self.transform != 'none':
                #
                augmentation = random.random()
                #
                if self.transform == 'flip':
                    #
                    if augmentation > 0.5:
                        #
                        image = np.flip(image, axis=1).copy()
                        image = np.flip(image, axis=2).copy()
                        label = np.flip(label, axis=1).copy()
                        label = np.flip(label, axis=2).copy()

                    return image, label, labelname

                elif self.transform == 'all':
                    #
                    if augmentation < 0.25:
                        # flip along x axis
                        image = np.flip(image, axis=1).copy()
                        label = np.flip(label, axis=1).copy()
                        image = np.flip(image, axis=2).copy()
                        label = np.flip(label, axis=2).copy()

                    elif augmentation < 0.5:
                        # change the channel ratio
                        channel_ratio = 0.8
                        # for channel in range(c):
                        image = image * channel_ratio

                    elif augmentation < 0.75:
                        # random Gaussian noises
                        mean = 0.0
                        sigma = 0.15
                        noise = np.random.normal(mean, sigma, image.shape)
                        mask_overflow_upper = image + noise >= 1.0
                        mask_overflow_lower = image + noise < 0.0
                        noise[mask_overflow_upper] = 1.0
                        noise[mask_overflow_lower] = 0.0
                        image += noise

                    return image, label, labelname

                elif self.transform == 'mixup':
                    #
                    alpha = 0.2
                    lam = np.random.beta(alpha, alpha)
                    #
                    another_index = random.randint(0, self.__len__() - 1)
                    another_image = tiff.imread(all_images[another_index])
                    another_image = np.array(another_image, dtype='float32')
                    another_label = tiff.imread(all_labels[another_index])
                    another_label = np.array(another_label, dtype='float32')
                    #
                    (height, width) = another_image.shape
                    another_label = another_label.reshape(1, height, width)
                    another_image = another_image.reshape(1, height, width)
                    #
                    mixed_image = lam * image + (1 - lam) * another_image
                    #
                    return image, label, labelname, another_image, another_label, mixed_image, lam
                #
            else:
                #
                return image, label, labelname

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.jpg')))


def evaluate(data, model, device, class_no):

    model.eval()

    with torch.no_grad():
        #
        f1 = 0
        test_iou = 0
        test_h_dist = 0
        recall = 0
        precision = 0
        #
        # for index in evaluate_index:
        for j, (testimg, testlabel, testimgname) in enumerate(data):

            # ===========================================================
            # ===========================================================

            testimg = testimg.to(device=device, dtype=torch.float32)

            testlabel = testlabel.to(device=device, dtype=torch.float32)

            testoutput = model(testimg)

            if class_no == 2:
                #
                testoutput = torch.sigmoid(testoutput)
                testoutput = (testoutput > 0.5).float()
                #
            else:
                #
                _, testoutput = torch.max(testoutput, dim=1)
                #
            # mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mean_iu_ = intersectionAndUnion(testoutput.cpu().detach(), testlabel.cpu().detach(), class_no)

            f1_, recall_, precision_ = f1_score(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            f1 += f1_
            test_iou += mean_iu_
            recall += recall_
            precision += precision_

    # return test_iou / len(evaluate_index), f1 / len(evaluate_index), recall / len(evaluate_index), precision / len(evaluate_index)
    return test_iou / (j + 1), f1 / (j + 1), recall / (j + 1), precision / (j + 1)


def test(data_1, data_2, model, device, class_no, save_location):

    model.eval()

    data_1_testoutputs = []
    data_2_testoutputs = []

    with torch.no_grad():
        #
        f1_1 = 0
        test_iou_1 = 0
        # test_h_dist_1 = 0
        recall_1 = 0
        precision_1 = 0
        mse_1 = 0
        #
        f1_2 = 0
        test_iou_2 = 0
        # test_h_dist_2 = 0
        recall_2 = 0
        precision_2 = 0
        mse_2 = 0
        #
        # ==============================================
        evaluate_index_all_1 = range(0, len(data_1) - 1)
        #
        # ==============================================
        evaluate_index_all_2 = range(0, len(data_2) - 1)
        #
        for j, (testimg, testlabel, testimgname) in enumerate(data_1):
            # extract a few random indexs every time in a range of the data
            # ========================================================================
            # ========================================================================
            #
            testimg = torch.from_numpy(testimg).to(device=device, dtype=torch.float32)
            #
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

                data_1_testoutputs.append(testoutput)
                #
            else:
                #
                _, testoutput = torch.max(testoutput_original, dim=1)
                #
            mean_iu_ = intersectionAndUnion(testoutput.cpu().detach(), testlabel.cpu().detach(), class_no)

            f1_, recall_, precision_ = f1_score(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mse_ = (np.square(testlabel.cpu().detach().numpy() - testoutput.cpu().detach().numpy())).mean()

            f1_1 += f1_
            test_iou_1 += mean_iu_
            recall_1 += recall_
            precision_1 += precision_
            mse_1 += mse_
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

        for j, (testimg, testlabel, testimgname) in enumerate(data_2):
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
                data_2_testoutputs.append(testoutput)
                #
            else:
                #
                _, testoutput = torch.max(testoutput_original, dim=1)
                data_2_testoutputs.append(testoutput)

            # mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mean_iu_ = intersectionAndUnion(testoutput.cpu().detach(), testlabel.cpu().detach(), class_no)

            f1_, recall_, precision_ = f1_score(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)

            mse_ = (np.square(testlabel.cpu().detach().numpy() - testoutput.cpu().detach().numpy())).mean()

            f1_2 += f1_
            test_iou_2 += mean_iu_
            recall_2 += recall_
            precision_2 += precision_
            mse_2 += mse_
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
    result_dictionary = {'Test IoU data 1': str(test_iou_1 / len(evaluate_index_all_1)),
                         'Test f1 data 1': str(f1_1 / len(evaluate_index_all_1)),
                         'Test recall data 1': str(recall_1 / len(evaluate_index_all_1)),
                         'Test Precision data 1': str(precision_1 / len(evaluate_index_all_1)),
                         'Test MSE data 1': str(mse_1 / len(evaluate_index_all_1)),
                         'Test IoU data 2': str(test_iou_2 / len(evaluate_index_all_2)),
                         'Test f1 data 2': str(f1_2 / len(evaluate_index_all_2)),
                         'Test recall data 2': str(recall_2 / len(evaluate_index_all_2)),
                         'Test Precision data 2': str(precision_2 / len(evaluate_index_all_2)),
                         'Test MSE data 2': str(mse_2 / len(evaluate_index_all_2)),}

    ff_path = prediction_map_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    return test_iou_1 / len(evaluate_index_all_1), f1_1 / len(evaluate_index_all_1), recall_1 / len(evaluate_index_all_1), precision_1 / len(evaluate_index_all_1), mse_1 / len(evaluate_index_all_1), \
           test_iou_2 / len(evaluate_index_all_2), f1_2 / len(evaluate_index_all_2), recall_2 / len(evaluate_index_all_2), precision_2 / len(evaluate_index_all_2), mse_2 / len(evaluate_index_all_2), \
           data_1_testoutputs, data_2_testoutputs


class EWC(object):
    def __init__(self, model, dataset, device, sample_size):

        self.model = model
        self.dataset = dataset
        self.device = device
        self.sample_size = sample_size

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            #
            self._means[n] = p.data.clone()

    def _diag_fisher(self):
        #
        precision_matrices = {}
        #
        for n, p in deepcopy(self.params).items():
            #
            p.data.zero_()
            precision_matrices[n] = p.data.clone()

        self.model.eval()

        for index, (input, label, input_name) in enumerate(self.dataset):
            #
            if index < self.sample_size:
                #
                self.model.zero_grad()
                #
                input = input.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)
                #
                output = self.model(input)
                # We use empirical Fisher here:
                loss = F.binary_cross_entropy_with_logits(F.logsigmoid(output), label)
                loss.backward()
                #
                for index, (n, p) in enumerate(self.model.named_parameters()):
                    #
                    precision_matrices[n].data += (p.grad.data**2).mean(0)

        precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss