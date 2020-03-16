from OCT_train import trainModels
# torch.manual_seed(0)
# ===================================================================================

if __name__ == '__main__':
    #
    # ===========================
    # U-Net based
    # ===========================
    # trainModels(model='SOASNet',
    #             input_dim=1,
    #             epochs=50,
    #             width=16,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=3,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=8,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='dice',
    #             norm='bn',
    #             log='MICCAI_2020_03_14',
    #             class_no=2,
    #             cluster=False)
    # # #
    # trainModels(model='unet',
    #             input_dim=1,
    #             epochs=50,
    #             width=16,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=3,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=8,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='dice',
    #             norm='bn',
    #             log='MICCAI_2020_03_14',
    #             class_no=2,
    #             cluster=False)
    # # #
    # trainModels(model='SOASNet_multi_attn',
    #             input_dim=1,
    #             epochs=50,
    #             width=16,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=3,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=8,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='dice',
    #             norm='bn',
    #             log='MICCAI_2020_03_14',
    #             class_no=2,
    #             cluster=False)
    # #
    # ====================================
    # SegNet based
    # ====================================
    #
    trainModels(model='SOASNet_very_large_kernel',
                data_set='ours',
                input_dim=1,
                epochs=1,
                width=16,
                depth=4,
                depth_limit=6,
                repeat=1,
                l_r=1e-3,
                l_r_s=True,
                train_batch=4,
                shuffle=True,
                loss='ce',
                norm='bn',
                log='Test',
                class_no=2,
                cluster=False,
                data_augmentation_train='all',
                data_augmentation_test='none')
    #
    # trainModels(model='RelayNet',
    #             input_dim=1,
    #             epochs=250,
    #             width=64,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=1,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=4,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='ce',
    #             norm='bn',
    #             log='MICCAI_2020_03_15',
    #             class_no=8,
    #             cluster=False)
    # #
    # trainModels(model='SOASNet_segnet',
    #             input_dim=1,
    #             epochs=250,
    #             width=64,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=1,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=4,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='ce',
    #             norm='bn',
    #             log='MICCAI_2020_03_15',
    #             class_no=8,
    #             cluster=False)
    # #
    # trainModels(model='SOASNet_segnet_skip',
    #             input_dim=1,
    #             epochs=250,
    #             width=64,
    #             depth=4,
    #             depth_limit=6,
    #             repeat=1,
    #             l_r=1e-3,
    #             l_r_s=True,
    #             train_batch=4,
    #             shuffle=True,
    #             data_augmentation='all',
    #             loss='ce',
    #             norm='bn',
    #             log='MICCAI_2020_03_15',
    #             class_no=8,
    #             cluster=False)
    #

print('End')