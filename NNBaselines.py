import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels, step, norm):
    #
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===================
    if norm == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.ReLU(inplace=True)
        )


def first_conv(in_channels, out_channels):
    # ====================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # ===================
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class MultiScaleFPAttenEncoder(nn.Module):
    #
    def __init__(self, in_channels, out_channels, step, norm):
        super(MultiScaleFPAttenEncoder, self).__init__()
        # ==============================================
        # in_channels: dimension of input
        # out_channels: dimension of output
        # step: stride
        # ==============================================
        # Attention branch:
        # Dilated attention with parameter sharing
        # Depth-wise
        # Default-setting:
        # To get rid of grid artifacts:
        # We use multi-scale dilation
        # ==========================
        dilation = 3
        # dilation performs pretty well
        channel_expansion = 4
        # channel expansion has to be larger than at least 3
        #
        self.attention_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, channel_expansion*out_channels, kernel_size=1, stride=step, bias=True),
        )
        #
        if norm == 'in':
            self.attention_branch_2_1 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, groups=channel_expansion*out_channels),
                nn.InstanceNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_2 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation, dilation=dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.InstanceNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_3 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation*dilation, dilation=dilation*dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.InstanceNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            # Trunk branch
            self.trunk_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        elif norm == 'bn':
            self.attention_branch_2_1 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, groups=channel_expansion*out_channels),
                nn.BatchNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_2 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation, dilation=dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.BatchNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_3 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation*dilation, dilation=dilation*dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.BatchNorm2d(channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            # Trunk branch
            self.trunk_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        elif norm == 'ln':
            self.attention_branch_2_1 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_2 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation, dilation=dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_3 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation*dilation, dilation=dilation*dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            # Trunk branch
            self.trunk_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
                nn.GroupNorm(out_channels, out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
                nn.GroupNorm(out_channels, out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
        elif norm == 'gn':
            self.attention_branch_2_1 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels // 8, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_2 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation, dilation=dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels // 8, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            #
            self.attention_branch_2_3 = nn.Sequential(
                nn.Conv2d(channel_expansion*out_channels, channel_expansion*out_channels, kernel_size=3, stride=1, padding=dilation*dilation*dilation, dilation=dilation*dilation*dilation, bias=False, groups=channel_expansion*out_channels),
                nn.GroupNorm(channel_expansion*out_channels // 8, channel_expansion*out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
            # Trunk branch
            self.trunk_branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
                nn.GroupNorm(out_channels // 8, out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
                nn.GroupNorm(out_channels // 8, out_channels, affine=True),
                nn.ReLU(inplace=True)
            )
        #
        self.attention_branch_3 = nn.Sequential(
            nn.Conv2d(channel_expansion*out_channels, out_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        #

    def forward(self, x):

        trunk_features = self.trunk_branch(x)

        attention_weights = self.attention_branch_1(x)
        attention_weights = self.attention_branch_2_1(attention_weights) + self.attention_branch_2_2(attention_weights) + self.attention_branch_2_3(attention_weights)
        attention_weights = self.attention_branch_2_1(attention_weights) + self.attention_branch_2_2(attention_weights) + self.attention_branch_2_3(attention_weights)
        attention_weights = self.attention_branch_3(attention_weights)

        output = trunk_features * attention_weights + trunk_features

        return output, attention_weights, trunk_features


def last_conv(in_channels, no_class, mode):
    #
    if mode == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Conv2d(in_channels, no_class, 1, stride=1, padding=0, bias=True)
        )

    elif mode == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.Conv2d(in_channels, no_class, 1, stride=1, padding=0, bias=True)
        )

    elif mode == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(in_channels, in_channels, affine=True),
            nn.Conv2d(in_channels, no_class, 1, stride=1, padding=0, bias=True)
        )

    elif mode == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(in_channels // 8, in_channels, affine=True),
            nn.Conv2d(in_channels, no_class, 1, stride=1, padding=0, bias=True)
        )


class segnet_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(segnet_encoder, self).__init__()
        self.convs_block = conv_block(in_channels, out_channels, step=1, norm=mode)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.convs_block(inputs)
        unpooled_size = outputs.size()
        outputs, indices = self.maxpool(outputs)
        return outputs, indices, unpooled_size


class segnet_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(segnet_decoder, self).__init__()
        self.convs_block = conv_block(in_channels, out_channels, step=1, norm=mode)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.convs_block(outputs)
        return outputs


class skip_connection(nn.Module):

    def __init__(self, in_channels, out_channels, s, norm):
        super(skip_connection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, bias=False)
        if norm == 'bn':
            self.smooth = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'in':
            self.smooth = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.smooth = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm == 'gn':
            self.smooth = nn.GroupNorm(out_channels // 8, out_channels, affine=True)

    def forward(self, inputs):
        output = self.smooth(self.conv(inputs))
        return output


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, step, norm):
        super(conv_block, self).__init__()
        #
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=step, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_1 = nn.ReLU(inplace=True)
        self.activation_2 = nn.ReLU(inplace=True)
        #
        if norm == 'bn':
            self.smooth_1 = nn.BatchNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'in':
            self.smooth_1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.smooth_1 = nn.GroupNorm(out_channels, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm == 'gn':
            self.smooth_1 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)

    def forward(self, inputs):
        output = self.activation_1(self.smooth_1(self.conv_1(inputs)))
        output = self.activation_2(self.smooth_2(self.conv_2(output)))
        return output


class conv_layer(nn.Module):

    def __init__(self, in_channels, out_channels, step, norm):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=step, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        if norm == 'bn':
            self.smooth = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'in':
            self.smooth = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.smooth = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm == 'gn':
            self.smooth = nn.GroupNorm(out_channels // 8, out_channels, affine=True)

    def forward(self, inputs):
        output = self.activation(self.smooth(self.conv(inputs)))
        return output


# =========================================================================
# Pi_model
# =========================================================================
# We implemented according to the description in appendix b of mean-teacher
# https://arxiv.org/pdf/1703.01780.pdf

class Pi_model(nn.Module):

    def __init__(self, in_ch, width, mode, n_classes, side_output):
        super(Pi_model, self).__init__()
        #
        self.side_output_mode = side_output
        #
        if n_classes == 2:
            output_channel = 1
        else:
            output_channel = n_classes
        #
        self.down_1 = conv_layer(in_ch, width, step=1, norm=mode)
        self.down_2 = conv_layer(width, width, step=1, norm=mode)
        self.down_3 = conv_layer(width, 2*width, step=1, norm=mode)
        self.pool_1 = nn.MaxPool2d(2)
        self.drop_1 = nn.Dropout2d(0.5)
        #
        self.down_4 = conv_layer(2*width, 2*width, step=1, norm=mode)
        self.down_5 = conv_layer(2*width, 2*width, step=1, norm=mode)
        self.down_6 = conv_layer(2*width, 4*width, step=1, norm=mode)
        self.pool_2 = nn.MaxPool2d(2)
        self.drop_2 = nn.Dropout2d(0.5)
        #
        self.down_7 = conv_layer(4*width, 2*width, step=1, norm=mode)
        self.down_8 = conv_layer(2*width, width, step=1, norm=mode)
        self.down_9 = conv_layer(width, width, step=1, norm=mode)
        #
        self.pool_3 = nn.AdaptiveAvgPool2d(1)
        #
        self.last_layer = nn.Linear(in_features=width, out_features=output_channel, bias=True)

    def forward(self, x):
        #
        output = self.down_1(x)
        output = self.down_2(output)
        output = self.down_3(output)
        output = self.drop_1(self.pool_1(output))
        #
        output = self.down_4(output)
        output = self.down_5(output)
        output = self.down_6(output)
        output = self.drop_2(self.pool_2(output))
        #
        output = self.down_7(output)
        output = self.down_8(output)
        output = self.down_9(output)
        output = self.pool_3(output)
        #
        output = self.last_layer(output)
        #
        return output


# ==================================


class MSFPAttentionUNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, mode, norm, dropout=False, visulisation=False, side_output=False):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages
        # mode: 'encoder', 'decoder', 'full', 'baseline'
        # visulisation: if it is true, it output attention weights for later visulisation
        # ===============================================================================
        super(MSFPAttentionUNet, self).__init__()
        #
        self.side_output_mode = side_output
        self.depth = depth
        self.atten_visual = visulisation
        self.dropout = dropout
        self.mode = mode
        #
        if class_no > 2:
            #
            self.final_in = class_no
        else:
            #
            self.final_in = 1
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        #
        if self.side_output_mode is True:
            #
            self.embeddings_small = nn.ModuleList()
            self.embeddings_large = nn.ModuleList()
        #
        if self.dropout is True:

            self.dropout_layers = nn.ModuleList()

        for i in range(self.depth):

            if self.side_output_mode is True:
                #
                self.embeddings_small.append(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.embeddings_large.append(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False))

            if self.dropout is True:

                self.dropout_layers.append(nn.Dropout2d(0.4))

            if i == 0:
                #
                self.encoders.append(first_conv(in_ch, width))
                #
                if mode == 'encoder' or mode == 'baseline':
                    self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                elif mode == 'decoder' or mode == 'full':
                    self.decoders.append(MultiScaleFPAttenEncoder(in_channels=width * 2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                if mode == 'encoder':
                    self.encoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                    self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'decoder':
                    self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                    self.decoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'full':
                    self.encoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                    self.decoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'baseline':
                    #
                    self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                    self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                    #
            else:
                #
                if mode == 'encoder':
                    self.encoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                    self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'decoder':
                    self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                    self.decoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'full':
                    self.encoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                    self.decoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                elif mode == 'baseline':
                    self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                    self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        if mode == 'encoder' or mode == 'full':
            self.encoders.append(MultiScaleFPAttenEncoder(in_channels=width*(2**(self.depth-2)), out_channels=width*(2**(self.depth-2)), step=2, norm=norm))
        elif mode == 'decoder' or mode == 'baseline':
            self.encoders.append(double_conv(in_channels=width*(2**(self.depth - 2)), out_channels=width* (2 ** (self.depth - 2)), step=2, norm=norm))
            #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        #
        if self.atten_visual is True:
            attention_weights = []
            trunk_features = []
        #
        if self.side_output_mode is True:
            side_embeddings = []
        #
        for i in range(len(self.encoders)):
            #
            if i == 0:
                #
                y = self.encoders[i](y)
                #
                encoder_features.append(y)
                #
            else:
                #
                if self.mode == 'baseline':
                    #
                    y = self.encoders[i](y)
                    #
                else:
                    #
                    y, a, t = self.encoders[i](y)
                    #
                    if i < (len(self.encoders) - 1):
                        #
                        encoder_features.append(y)
                    #
                    if self.atten_visual is True:
                        #
                        a = torch.mean(a, dim=1, keepdim=True)
                        t = torch.mean(t, dim=1, keepdim=True)
                        #
                        attention_weights.append(a)
                        trunk_features.append(t)
        #
        for i in range(len(encoder_features)):
            #
            # print(i)
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
            diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[-(i+1)](y)
            #
            if self.dropout is True:
                #
                y = self.dropout_layers[i](y)
            #
            if self.side_output_mode is True:
                #
                max_rep, __ = torch.max(y, dim=1, keepdim=True)
                avg_rep = torch.mean(y, dim=1, keepdim=True)
                #
                task_representation = torch.cat([max_rep, avg_rep], dim=1)
                task_representation = self.embeddings_small[i](task_representation) + self.embeddings_large[i](task_representation)
                #
                side_embeddings.append(task_representation / 2)
        #
        y = self.conv_last(y)
        #
        if self.atten_visual is False and self.side_output_mode is False:
            #
            return y
        #
        elif self.atten_visual is False and self.side_output_mode is True:
            #
            return y, side_embeddings
        #
        elif self.atten_visual is True and self.side_output_mode is True:
            #
            return y, side_embeddings, attention_weights, trunk_features
        #
        elif self.atten_visual is True and self.side_output_mode is False:
            #
            return y, attention_weights, trunk_features

# =====================================================


class SegNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, n_classes, dropout, side_output, norm='bn'):
        #
        super(SegNet, self).__init__()
        #
        self.side_output_mode = side_output
        self.dropout_mode = dropout
        self.depth = depth
        #
        if n_classes == 2:

            output_channel = 1

        else:

            output_channel = n_classes

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        if self.side_output_mode is True:

            self.embeddings_small = nn.ModuleList()
            self.embeddings_large = nn.ModuleList()

        if self.dropout_mode is True:

            self.dropout_layers = nn.ModuleList()

        for i in range(self.depth):

            if i == 0:

                self.encoders.append((segnet_encoder(in_ch, width, norm)))
                self.decoders.append((segnet_decoder(width, width, norm)))

            else:

                self.encoders.append((segnet_encoder(width*(2**(i - 1)), width*(2**i), norm)))
                self.decoders.append((segnet_decoder(width*(2**i), width*(2**(i - 1)), norm)))

            if self.side_output_mode is True:

                self.embeddings_small.append(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.embeddings_large.append(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False))

            if self.dropout_mode is True:

                self.dropout_layers.append(nn.Dropout2d(0.4))

        self.classification_layer = nn.Conv2d(width, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        encoder_features = []
        encoder_indices = []
        encoder_pool_shapes = []

        if self.side_output_mode is True:

            side_outputs = []

        for i in range(len(self.encoders)):
            #
            x, indice, shape = self.encoders[i](x)
            #
            encoder_features.append(x)
            encoder_indices.append(indice)
            encoder_pool_shapes.append(shape)
            #
        for i in range(len(encoder_features)):
            #
            x = self.decoders[len(encoder_features) - i - 1](x, encoder_indices[len(encoder_features) - i - 1], encoder_pool_shapes[len(encoder_features) - i - 1])
            #
            if self.dropout_mode is True:
                #
                x = self.dropout_layers[i](x)
            #
            if self.side_output_mode is True:
                #
                max_rep, __ = torch.max(x, dim=1, keepdim=True)
                avg_rep = torch.mean(x, dim=1, keepdim=True)
                #
                task_representation = torch.cat([max_rep, avg_rep], dim=1)
                task_representation = self.embeddings_small[i](task_representation) + self.embeddings_large[i](task_representation)
                #
                side_outputs.append(task_representation / 2)
                #
        #
        x = self.classification_layer(x)
        #
        if self.side_output_mode is True:
            #
            return x, side_outputs
        else:
            #
            return x


# class ResidualUNet(nn.Module):
#     #
#     def __init__(self, in_ch, width, norm, n_classes, dropout, side_output):
#         #
#         super().__init__()
#         self.side_output_mode = side_output
#         self.dropout_mode = dropout
#         #
#         if n_classes == 2:
#             self.final_in = 1
#         else:
#             self.final_in = n_classes
#         #
#         self.w1 = width
#         self.w2 = width * 2
#         self.w3 = width * 4
#         self.w4 = width * 8
#         self.w5 = width * 16
#         #
#         self.dconv_down0 = conv_block(in_ch, self.w1, step=1, norm=norm)
#         self.dconv_down0_1 = conv_block(self.w1, self.w1, step=1, norm=norm)
#         self.dconv_down0_skip = skip_connection(in_channels=self.w1, out_channels=self.w1, s=1, norm=norm)
#         #
#         self.dconv_down1_1 = conv_block(self.w1, self.w2, step=2, norm=norm)
#         self.dconv_down1_2 = conv_block(self.w2, self.w2, step=1, norm=norm)
#         self.dconv_down1_skip = skip_connection(in_channels=self.w1, out_channels=self.w2, s=2, norm=norm)
#         #
#         self.dconv_down2_1 = conv_block(self.w2, self.w2, step=1, norm=norm)
#         self.dconv_down2_2 = conv_block(self.w2, self.w2, step=1, norm=norm)
#         self.dconv_down2_skip = skip_connection(in_channels=self.w2, out_channels=self.w2, s=1, norm=norm)
#         #
#         self.dconv_down3_1 = conv_block(self.w2, self.w3, step=2, norm=norm)
#         self.dconv_down3_2 = conv_block(self.w3, self.w3, step=1, norm=norm)
#         self.dconv_down3_skip = skip_connection(in_channels=self.w2, out_channels=self.w3, s=2, norm=norm)
#         #
#         self.dconv_down4_1 = conv_block(self.w3, self.w3, step=1, norm=norm)
#         self.dconv_down4_2 = conv_block(self.w3, self.w3, step=1, norm=norm)
#         self.dconv_down4_skip = skip_connection(in_channels=self.w3, out_channels=self.w3, s=1, norm=norm)
#         #
#         self.dconv_down5_1 = conv_block(self.w3, self.w3, step=2, norm=norm)
#         self.dconv_down5_2 = conv_block(self.w3, self.w3, step=1, norm=norm)
#         self.dconv_down5_skip = skip_connection(in_channels=self.w3, out_channels=self.w3, s=2, norm=norm)
#         #
#         self.dconv_down6_1 = conv_block(self.w3, self.w4, step=2, norm=norm)
#         self.dconv_down6_2 = conv_block(self.w4, self.w4, step=1, norm=norm)
#         self.dconv_down6_skip = skip_connection(in_channels=self.w3, out_channels=self.w4, s=2, norm=norm)
#         #
#         self.dconv_down7_1 = conv_block(self.w4, self.w4, step=2, norm=norm)
#         self.dconv_down7_2 = conv_block(self.w4, self.w4, step=1, norm=norm)
#         self.bridge = conv_block(self.w4, self.w4, step=1, norm=norm)
#         self.dconv_down7_skip = skip_connection(in_channels=self.w4, out_channels=self.w4, s=2, norm=norm)
#         #
#         self.dconv_up4 = conv_block(self.w4 + self.w4, self.w4, step=1, norm=norm)
#         self.dconv_up3 = conv_block(self.w4 + self.w3, self.w3, step=1, norm=norm)
#         self.dconv_up2 = conv_block(self.w3 + self.w3, self.w3, step=1, norm=norm)
#         self.dconv_up1 = conv_block(self.w3 + self.w2, self.w2, step=1, norm=norm)
#         self.dconv_up0 = conv_block(self.w2 + self.w1, self.w1, step=1, norm=norm)
#         #
#         self.max_pool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         #
#         self.classification_layer = nn.Conv2d(self.w1, self.final_in, kernel_size=1, stride=1, padding=0, bias=True)
#         #
#         if self.dropout_mode is True:
#             #
#             self.drop1 = nn.Dropout2d(0.5)
#             self.drop2 = nn.Dropout2d(0.4)
#             self.drop3 = nn.Dropout2d(0.3)
#             self.drop4 = nn.Dropout2d(0.2)
#             self.drop5 = nn.Dropout2d(0.1)
#             #
#         if self.side_output_mode is True:
#             #
#             self.task_embedding_1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.task_embedding_2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.task_embedding_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.task_embedding_4 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
#             self.task_embedding_5 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         conv0 = self.dconv_down0(x)
#         conv0 = self.dconv_down0_1(conv0) + self.dconv_down0_skip(conv0)
#         conv1 = self.dconv_down1_2(self.dconv_down1_1(conv0)) + self.dconv_down1_skip(conv0)
#         conv2 = self.dconv_down2_2(self.dconv_down2_1(conv1)) + self.dconv_down2_skip(conv1)
#         conv3 = self.dconv_down3_2(self.dconv_down3_1(conv2)) + self.dconv_down3_skip(conv2)
#         conv4 = self.dconv_down4_2(self.dconv_down4_1(conv3)) + self.dconv_down4_skip(conv3)
#         conv5 = self.dconv_down5_2(self.dconv_down5_1(conv4)) + self.dconv_down5_skip(conv4)
#         conv6 = self.dconv_down6_2(self.dconv_down6_1(conv5)) + self.dconv_down6_skip(conv5)
#         conv7 = self.bridge(self.dconv_down7_2(self.dconv_down7_1(conv6))) + self.dconv_down7_skip(conv6)
#         #
#         max_rep, __ = torch.max(conv7, dim=1, keepdim=True)
#         avg_rep = torch.mean(conv7, dim=1, keepdim=True)
#         task_representation_1 = torch.cat([max_rep, avg_rep], dim=1)
#         task_representation_1 = self.task_embedding_1(task_representation_1)
#         #
#         if self.dropout_mode is False:
#             x = self.upsample(conv7)
#             x = torch.cat([x, conv6], dim=1)
#             x = self.dconv_up4(x)
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_2 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_2 = self.task_embedding_2(task_representation_2)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv5], dim=1)
#             x = self.dconv_up3(x)
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_3 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_3 = self.task_embedding_3(task_representation_3)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv4], dim=1)
#             x = self.dconv_up2(x)
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_4 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_4 = self.task_embedding_4(task_representation_4)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv2], dim=1)
#             x = self.dconv_up1(x)
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_5 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_5 = self.task_embedding_5(task_representation_5)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv0], dim=1)
#             x = self.dconv_up0(x)
#             out = self.classification_layer(x)
#         else:
#             x = self.upsample(conv7)
#             x = torch.cat([x, conv6], dim=1)
#             x = self.drop1(self.dconv_up4(x))
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_2 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_2 = self.task_embedding_2(task_representation_2)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv5], dim=1)
#             x = self.drop2(self.dconv_up3(x))
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_3 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_3 = self.task_embedding_3(task_representation_3)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv4], dim=1)
#             x = self.drop3(self.dconv_up2(x))
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_4 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_4 = self.task_embedding_4(task_representation_4)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv2], dim=1)
#             x = self.drop4(self.dconv_up1(x))
#             #
#             if self.side_output_mode is True:
#                 max_rep, __ = torch.max(x, dim=1, keepdim=True)
#                 avg_rep = torch.mean(x, dim=1, keepdim=True)
#                 task_representation_5 = torch.cat([max_rep, avg_rep], dim=1)
#                 task_representation_5 = self.task_embedding_5(task_representation_5)
#             #
#             x = self.upsample(x)
#             x = torch.cat([x, conv0], dim=1)
#             x = self.drop5(self.dconv_up0(x))
#             out = self.classification_layer(x)
#
#         if self.side_output_mode is True:
#             return out, task_representation_1, task_representation_2, task_representation_3, task_representation_4, task_representation_5
#         else:
#             return out