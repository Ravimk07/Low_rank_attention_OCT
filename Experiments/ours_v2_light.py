import torch
import sys
sys.path.append("..")
# ===================
from OCT_train import trainModels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    #
    trainModels(model='TripleNet_v2',
                input_dim=1,
                epochs=300,
                width=8,
                depth=4,
                depth_limit=6,
                repeat=3,
                l_r=1e-3,
                l_r_s=True,
                train_batch=8,
                shuffle=True,
                data_augmentation='all',
                loss='dice',
                norm='in',
                log='MICCAI_2020',
                class_no=2,
                cluster=True)

    print('Finished.')