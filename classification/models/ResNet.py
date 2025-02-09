from models.layers import weights_init

import torch.nn as nn
# import torchvision.models
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

        pretrained = True
        if hasattr(FLAGS, 'pretrained'):
            self.pretrained = FLAGS.pretrained
        elif FLAGS.load_ckpt != '':
            pretrained = False

        if layers == 18:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)

        elif layers == 34:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet34(weights=weights)

        elif layers == 50:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)

        elif layers == 101:
            from torchvision.models import resnet101, ResNet101_Weights
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet101(weights=weights)

        elif layers == 152:
            from torchvision.models import resnet152, ResNet152_Weights
            weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet152(weights=weights)

        self.in_channels = FLAGS.num_channels
        if self.in_channels == 3:
            self.conv1 = model._modules['conv1']
            self.bn1 = model._modules['bn1']
        else:
            in_channels_conv1 = 64
            self.conv1 = nn.Conv2d(self.in_channels, in_channels_conv1, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels_conv1)

        self.relu = model._modules['relu']
        self.maxpool = model._modules['maxpool']
        self.layer1 = model._modules['layer1']
        self.layer2 = model._modules['layer2']
        self.layer3 = model._modules['layer3']
        self.layer4 = model._modules['layer4']

        # clear memory
        del model


    def init_weights(self):
        if self.in_channels == 3:
            weights_init(self.conv1)
            weights_init(self.bn1)

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
        out = glob_pool.view(glob_pool.size(0), -1)
        return out
























