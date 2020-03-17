import torch
import sys
sys.path.append("..")
# ===================
from OCT_train import trainModels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


trainModels(model='SOASNet_large_kernel',
            data_set='duke',
            input_dim=1,
            epochs=250,
            width=64,
            depth=4,
            depth_limit=6,
            repeat=5,
            l_r=1e-3,
            l_r_s=True,
            train_batch=4,
            shuffle=True,
            loss='ce',
            norm='bn',
            log='MICCAI_Duke_Results',
            class_no=8,
            cluster=True,
            data_augmentation_train='all',
            data_augmentation_test='none')