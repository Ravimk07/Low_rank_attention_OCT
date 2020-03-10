from OCT_train import trainModels
# torch.manual_seed(0)
# ===================================================================================

if __name__ == '__main__':
    #
    # trainModels(model='TripleNet',
    #             input_dim=1,
    #             epochs=200,
    #             width=16,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=2,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=8,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='dice',
    #             norm='in',
    #             log='MICCAI_2020_03_09',
    #             class_no=2,
    #             cluster=False)
    #
    # trainModels(model='TripleNet_v2',
    #             input_dim=1,
    #             epochs=200,
    #             width=16,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=2,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=8,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='dice',
    #             norm='in',
    #             log='MICCAI_2020_03_09',
    #             class_no=2,
    #             cluster=False)
    #
    trainModels(model='unet',
                input_dim=1,
                epochs=200,
                width=16,
                depth=4,
                depth_limit=6,
                repeat=2,
                l_r=1e-3,
                l_r_s=True,
                train_batch=8,
                shuffle=True,
                data_augmentation='all',
                loss='dice',
                norm='in',
                log='MICCAI_2020_03_09',
                class_no=2,
                cluster=False)
    #
    trainModels(model='Segnet',
                input_dim=1,
                epochs=200,
                width=16,
                depth=4,
                depth_limit=6,
                repeat=2,
                l_r=1e-3,
                l_r_s=True,
                train_batch=8,
                shuffle=True,
                data_augmentation='all',
                loss='dice',
                norm='in',
                log='MICCAI_2020_03_09',
                class_no=2,
                cluster=False)
    #
print('End')