# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput, nb_tasks=1):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn_ini = nn.ModuleList(
            [nn.BatchNorm2d(noutput, eps=1e-3) for i in range(nb_tasks)])

    def forward(self, input):
        task = current_task
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn_ini[task](output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class non_bottleneck_1d_RAP (nn.Module):
    def __init__(self, chann, dropprob, dilated, nb_tasks=1):
        # chann = #channels, dropprob=dropout probability, dilated=dilation rate
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        # domain-specific 1x1conv
        self.parallel_conv_1 = nn.ModuleList([nn.Conv2d(chann, chann, kernel_size=1, stride=1, padding=0, bias=True) for i in range(
            nb_tasks)])  # nb_tasks=1 for 1st time, its only on CS
        self.bns_1 = nn.ModuleList(
            [nn.BatchNorm2d(chann, eps=1e-03) for i in range(nb_tasks)])

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.parallel_conv_2 = nn.ModuleList([nn.Conv2d(
            chann, chann, kernel_size=1, stride=1, padding=0, bias=True) for i in range(nb_tasks)])
        self.bns_2 = nn.ModuleList(
            [nn.BatchNorm2d(chann, eps=1e-03) for i in range(nb_tasks)])

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        task = current_task
        # print('input: ', input.size())
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        # print('output 2nd 1x3: ', output.size())

        # RAP skip connection for conv2
        output = output + self.parallel_conv_1[task](input)
        output = self.bns_1[task](output)

        output_ = F.relu(output)

        output = self.conv3x1_2(output_)
        output = F.relu(output)
        output = self.conv1x3_2(output)

        # RAP skip connection for conv2
        output = output + self.parallel_conv_2[task](output_)
        output = self.bns_2[task](output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


'''
ENCODER will use the non_bottleneck_1d_RAP modules as they have the parallel residual adapters.
DECODER will use non_bottleneck_1d modules as they don't have RAPs and we need RAPs only in the encoder.

only encoder has shared and domain-specific RAPs. decoder is domain specific
it'll be like decoder.0, decoder.1
for domain-specific RAPs and bns, it'll be like parallel_conv_2.0.weight, parallel_conv_2.1.weight
'''


class Encoder(nn.Module):
    def __init__(self, nb_tasks=1):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16, nb_tasks)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64, nb_tasks))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d_RAP(64, 0.03, 1, nb_tasks))

        self.layers.append(DownsamplerBlock(64, 128, nb_tasks))

        for x in range(0, 2):  # 2 times
            # dropprob for imagenet pretrained encoder is 0.1 not 0.3, here using 0.3 for imagenet pretrained encoder
            self.layers.append(non_bottleneck_1d_RAP(128, 0.3, 2, nb_tasks))
            self.layers.append(non_bottleneck_1d_RAP(128, 0.3, 4, nb_tasks))
            self.layers.append(non_bottleneck_1d_RAP(128, 0.3, 8, nb_tasks))
            self.layers.append(non_bottleneck_1d_RAP(128, 0.3, 16, nb_tasks))

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2,
                                       padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class Net(nn.Module):
    # use encoder to pass pretrained encoder
    def __init__(self, num_classes=[13], nb_tasks=1, cur_task=0):
        # the encoder has been passed here in this manner because we needed an Imagenet pretrained encoder. so used erfnet_imagenet to initialize encoder and read it from saved pretrained model. we want to attach that encoder to our decoder
        # encoder is not being passed. figure out another way of initialising with the imagenet pretrained encoder weights, on this encoder. init to be handled.
        super().__init__()

        global current_task
        current_task = cur_task

        self.encoder = Encoder(nb_tasks)

        self.decoder = nn.ModuleList(
            [Decoder(num_classes[i]) for i in range(nb_tasks)])

    def forward(self, input, task):
        global current_task
        # chose which branch of forward pass you need based on if training on current dataset or validating on a previous dataset.
        current_task = task
        output = self.encoder(input)
        output = self.decoder[task].forward(output)
        return output
