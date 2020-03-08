import os
import errno
import torch
import torch.nn as nn
import numpy as np
import time
import timeit
from torch.utils import data
import torch.nn.functional as F

from torch.optim import lr_scheduler
from NNLoss import dice_loss
from NNMetrics import segmentation_scores, f1_score
from NNUtils import evaluate, test
from tensorboardX import SummaryWriter
from torch.autograd import grad
# ================================================
from NNBaselines import SegNet
from SecondOrderAttentionStripNet import SOASNet
from adamW import AdamW
# =============================
from NNUtils import getData_OCT
# =============================


def trainModels(repeat, input_dim, task_order, train_batch, model, epochs, width, l_r, l_r_s, shuffle, data_augmentation, loss, norm, log, class_no, depth, cluster=False):
    #
    if cluster is False:
        #
        data_directory = '/home/moucheng/projects_data/OCT/'
        #
    else:
        #
        data_directory = '/cluster/project9/CKT_Project/projects_data/OCT/'
        #
    trainloader, train_dataset, validate_dataset, test_dataset = getData_OCT(data_directory, train_batch, shuffle_mode=shuffle, augmentation=data_augmentation)
    #
    for j in range(1, repeat+1, 1):
        #
        trained_model = trainSingleModel(model_name=model,
                                         epochs=epochs,
                                         width=width,
                                         lr=l_r,
                                         repeat=str(j),
                                         lr_scedule=l_r_s,
                                         task_order=task_order,
                                         train_dataset=train_dataset,
                                         train_batch=train_batch,
                                         train_loader=trainloader,
                                         validate_data=validate_dataset,
                                         test_data=test_dataset,
                                         data_augmentation=data_augmentation,
                                         shuffle=shuffle,
                                         loss=loss,
                                         norm=norm,
                                         log=log,
                                         no_class=class_no,
                                         input_channel=input_dim,
                                         depth=depth)


def trainSingleModel(model_name,
                     epochs,
                     width,
                     depth,
                     repeat,
                     lr,
                     lr_scedule,
                     task_order,
                     train_dataset,
                     train_batch,
                     data_augmentation,
                     train_loader,
                     validate_data,
                     test_data,
                     shuffle,
                     loss,
                     norm,
                     log,
                     no_class,
                     input_channel):
    # :param model: network module
    # :param epochs: training total epochs
    # :param width: first encoder channel number
    # :param lr: learning rate
    # :param lr_scedule: true or false for learning rate schedule
    # :param repeat: repeat same experiments
    # :param train_dataset: training data set
    # :param train_batch: batch size
    # :param train_loader: training loader
    # :param validate_loader: validation loader
    # :param shuffle: shuffle training data or not
    # :param loss: loss function tag, use 'ce' for cross-entropy
    # :param weights_transfer: 'dynamic', 'static' or 'average'
    # :param alpha: weight for knowledge distillation loss
    # :param norm_1: normalisation for model 1
    # :param norm_2: normalisation for model 2
    # :param log: log tag for recording experiments
    # :param no_class: 2 or multi-class
    # :param input_channel: 4 for BRATS, 3 for CityScapes
    # :param dataset_name: name of the dataset
    # :param temperature_start: 2 or 4
    # :param temperature_end: 4 or 2
    # :return:
    device = torch.device('cuda:0')

    # side_output_use = False

    if model_name == 'unet':

        model = SOASNet(in_ch=input_channel, width=width, depth=depth, norm=norm, n_classes=no_class, mode='unet', side_output=False, downsampling_limit=6).to(device=device)

    elif model_name == 'Segnet':

        model = SegNet(in_ch=input_channel, width=width, norm=norm, depth=depth, n_classes=no_class, dropout=True, side_output=False).to(device=device)

    elif model_name == 'TripleNet':

        model = SOASNet(in_ch=input_channel, width=width, depth=depth, norm=norm, n_classes=no_class, mode='low_rank_attn', side_output=False, downsampling_limit=3).to(device=device)

    # ==================================
    training_amount = len(train_dataset)
    iteration_amount = training_amount // train_batch
    iteration_amount = iteration_amount - 1

    model_name = model_name + '_Epoch_' + str(epochs) + \
                 '_Batch_' + str(train_batch) + \
                 '_Width_' + str(width) + \
                 '_Loss_' + loss + \
                 '_Norm_' + norm + \
                 '_ShuffleTraining_' + str(shuffle) + \
                 '_Data_Augmentation_' + data_augmentation + '_' + \
                 '_lr_' + str(lr) + \
                 '_Task_order_' + task_order + \
                 '_Repeat_' + str(repeat)

    print(model_name)

    writer = SummaryWriter('../../Log_' + log + '/' + model_name)

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # if lr_scedule is True:
    #     learning_rate_steps = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        # i: index of mini batch
        if 'mixup' not in data_augmentation:

            for j, (images, labels, imagename) in enumerate(train_loader):

                images = images.to(device=device, dtype=torch.float32)

                if no_class == 2:

                    labels = labels.to(device=device, dtype=torch.float32)

                else:

                    labels = labels.to(device=device, dtype=torch.long)

                outputs_logits = model(images)

                optimizer.zero_grad()

                # calculate main losses for second time
                if no_class == 2:
                    #
                    if loss == 'dice':
                        #
                        main_loss = dice_loss(torch.sigmoid(outputs_logits), labels)
                        #
                    elif loss == 'ce':
                        #
                        main_loss = nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)
                        #
                    elif loss == 'hybrid':
                        #
                        main_loss = dice_loss(torch.sigmoid(outputs_logits), labels) + nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)

                else:

                    main_loss = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(outputs_logits, dim=1), labels.squeeze(1))

                running_loss += main_loss

                main_loss.backward()

                optimizer.step()

                # ==============================================================================
                # Calculate training and validation metrics at the last iteration of each epoch
                # ==============================================================================

                if (j + 1) % iteration_amount == 0:

                    if no_class == 2:

                        outputs = torch.sigmoid(outputs_logits)

                        outputs = (outputs > 0.5).float()

                    else:

                        _, outputs = torch.max(outputs_logits, dim=1)

                        outputs = outputs.unsqueeze(1)

                    mean_iu = segmentation_scores(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), no_class)

                    validate_iou, validate_f1, validate_recall, validate_precision = evaluate(data=validate_data, model=model, device=device, class_no=no_class)

                    print(
                        'Step [{}/{}], '
                        'loss: {:.5f}, '
                        'train iou: {:.5f}, '
                        'val iou: {:.5f}'.format(epoch + 1,
                                                 epochs,
                                                 running_loss / (j + 1),
                                                 mean_iu,
                                                 validate_iou))

                    writer.add_scalars('scalars', {'train iou': mean_iu,
                                                   'val iou': validate_iou,
                                                   'val f1': validate_f1,
                                                   'val recall': validate_recall,
                                                   'val precision': validate_precision}, epoch + 1)

        else:
            # mix-up strategy requires more calculations:

            for j, (images_1, labels_1, imagename_1, images_2, labels_2, mixed_up_image, lam) in enumerate(train_loader):

                mixed_up_image = mixed_up_image.to(device=device, dtype=torch.float32)
                lam = lam.to(device=device, dtype=torch.float32)

                if no_class == 2:
                    labels_1 = labels_1.to(device=device, dtype=torch.float32)
                    labels_2 = labels_2.to(device=device, dtype=torch.float32)
                else:
                    labels_1 = labels_1.to(device=device, dtype=torch.long)
                    labels_2 = labels_2.to(device=device, dtype=torch.long)

                outputs_logits = model(mixed_up_image)

                optimizer.zero_grad()

                # calculate main losses for second time
                if no_class == 2:

                    if loss == 'dice':

                        main_loss = lam * dice_loss(torch.sigmoid(outputs_logits), labels_1) + (1 - lam) * dice_loss(torch.sigmoid(outputs_logits), labels_2)

                    elif loss == 'ce':

                        main_loss = lam * nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels_1) + (1 - lam) * nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels_2)

                    elif loss == 'hybrid':

                        main_loss = lam * dice_loss(torch.sigmoid(outputs_logits), labels_1) \
                                    + (1 - lam) * dice_loss(torch.sigmoid(outputs_logits), labels_2) \
                                    + lam * nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels_1) \
                                    + (1 - lam) * nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels_2)

                elif no_class == 8:

                    main_loss = lam * nn.CrossEntropyLoss(reduction='mean')(outputs_logits, labels_1.squeeze(1)) + (1 - lam) * nn.CrossEntropyLoss(reduction='mean')(outputs_logits, labels_2.squeeze(1))

                else:
                    main_loss = lam * nn.CrossEntropyLoss(reduction='mean')(outputs_logits, labels_1.squeeze(1)) + (1 - lam) * nn.CrossEntropyLoss(reduction='mean')(outputs_logits, labels_2.squeeze(1))

                running_loss += main_loss.mean()

                main_loss.mean().backward()

                optimizer.step()

                # ==============================================================================
                # Calculate training and validation metrics at the last iteration of each epoch
                # ==============================================================================
                if (j + 1) % iteration_amount == 0:

                    if no_class == 2:

                        outputs = torch.sigmoid(outputs_logits)

                    else:

                        _, outputs = torch.max(outputs_logits, dim=1)

                        outputs = outputs.unsqueeze(1)

                    mean_iu_1 = segmentation_scores(labels_1.cpu().detach().numpy(), outputs.cpu().detach().numpy(), no_class)

                    mean_iu_2 = segmentation_scores(labels_2.cpu().detach().numpy(), outputs.cpu().detach().numpy(), no_class)

                    mean_iu = lam.data.sum() * mean_iu_1 + (1 - lam.data.sum()) * mean_iu_2

                    validate_iou, validate_f1, validate_recall, validate_precision = evaluate(data=validate_data, model=model, device=device, class_no=no_class)

                    mean_iu = mean_iu.item()

                    print(
                        'Step [{}/{}], '
                        'loss: {:.4f}, '
                        'train iou: {:.4f}, '
                        'val iou: {:.4f}'.format(epoch + 1,
                                                 epochs,
                                                 running_loss / (j + 1),
                                                 mean_iu,
                                                 validate_iou))

                    writer.add_scalars('scalars', {'train iou': mean_iu,
                                                   'val iou': validate_iou,
                                                   'val f1': validate_f1,
                                                   'val recall': validate_recall,
                                                   'val precision': validate_precision}, epoch + 1)

        if lr_scedule is True:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr*((1 - epoch / epochs)**0.999)

    # save model
    save_folder = '../../saved_models/' + log

    try:
        os.makedirs(save_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    save_model_name = model_name + '_Final'

    save_model_name_full = save_folder + '/' + save_model_name + '.pt'

    torch.save(model, save_model_name_full)
    # =======================================================================
    # testing (disabled during training, because it is too slow)
    # =======================================================================
    save_results_folder = save_folder + '/testing_results_' + model_name

    try:
        os.makedirs(save_results_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    test_iou, test_f1, test_recall, test_precision = test(data=test_data, model=model, device=device, class_no=no_class, save_location=save_results_folder)

    print(
        'test iou: {:.4f}, '
        'test f1: {:.4f},'
        'test recall: {:.4f}, '
        'test precision: {:.4f}, '.format(test_iou,
                                          test_f1,
                                          test_recall,
                                          test_precision))

    print('\nTesting finished and results saved.\n')

    return save_model_name_full


