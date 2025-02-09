from collections import OrderedDict

from models.layers import weights_init

import torch.nn as nn
import torchvision.models
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, FLAGS):
        super(ResNet, self).__init__()
        layers = 18
        if hasattr(FLAGS, 'num_layers'):
            layers = FLAGS.num_layers
        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        self.batch_norm = False
        if hasattr(FLAGS, 'batch_norm'):
            self.batch_norm = FLAGS.batch_norm
        self.keep_prob = 1.0
        if hasattr(FLAGS, 'keep_prob'):
            self.keep_prob = FLAGS.keep_prob
        self.num_classes = FLAGS.num_classes

        pretrained = True
        if hasattr(FLAGS, 'pretrained'):
            self.pretrained = FLAGS.pretrained
        elif FLAGS.load_ckpt != '':
            pretrained = False
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        self.in_channels = FLAGS.num_channels
        if self.in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            in_channels_conv1 = 64
            self.conv1 = nn.Conv2d(self.in_channels, in_channels_conv1, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels_conv1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of ResNet output channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        # FCs
        layers_fc = OrderedDict()
        self.fc = list(map(int, FLAGS.fc.split(',')))
        current_dims = num_channels
        for i, n in enumerate(self.fc):
            layers_fc['fc{}'.format(i + 1)] = nn.Linear(current_dims, n)
            if self.batch_norm:
                layers_fc['bn{}'.format(i + 1)] = nn.BatchNorm1d(n)
            layers_fc['relu{}'.format(i + 1)] = nn.ReLU()
            layers_fc['drop{}'.format(i + 1)] = nn.Dropout(1 - self.keep_prob)
            current_dims = n
        self.fcs = nn.Sequential(layers_fc)

        self.classifier = nn.Sequential(nn.Linear(current_dims, self.num_classes))


    def init_weights(self):
        if self.in_channels == 3:
            weights_init(self.conv1)
            weights_init(self.bn1)
            weights_init(self.fcs)


    def forward(self, image):
        conv1 = self.conv1(image)
        conv1_bn = self.bn1(conv1)
        conv1_bn_relu = self.relu(conv1_bn)
        pool1 = self.maxpool(conv1_bn_relu)
        layer1 = self.layer1(pool1)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        resnet_out = self.layer4(layer3)

        # global average pooling
        glob_pool = F.adaptive_avg_pool2d(resnet_out, (1, 1))
        glob_pool = glob_pool.view(glob_pool.size(0), -1)

        fcs = self.fcs.forward(glob_pool)

        out = self.classifier(fcs)

        return out























