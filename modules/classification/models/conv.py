from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self, FLAGS):
        super(conv, self).__init__()
        self.num_channels = FLAGS.num_channels
        self.n_filters = list(map(int, FLAGS.n_filters.split(',')))
        self.fc = []
        if hasattr(FLAGS, 'fc'):
            self.fc = list(map(int, FLAGS.fc.split(',')))
        self.stride = [1]*len(self.n_filters)
        if hasattr(FLAGS, 'fc'):
            self.stride = list(map(int, FLAGS.stride.split(',')))
            assert len(self.stride) == len(self.n_filters)
        self.num_classes = FLAGS.num_classes
        self.batch_norm = False
        if hasattr(FLAGS, 'batch_norm'):
            self.batch_norm = FLAGS.batch_norm
        self.keep_prob = 1.0
        if hasattr(FLAGS, 'keep_prob'):
            self.keep_prob = FLAGS.keep_prob
        self.kernel_size = 3
        if hasattr(FLAGS, 'kernel_size'):
            self.kernel_size = int(FLAGS.kernel_size)


        # convolusions
        layers = OrderedDict()
        current_dims = self.num_channels
        for i, n_filter in enumerate(self.n_filters):
            layers['conv{}'.format(i+1)] = nn.Conv2d(current_dims, n_filter, kernel_size=self.kernel_size,
                                                     stride=self.stride[i], padding=0)
            if self.batch_norm:
                layers['bn{}'.format(i + 1)] = nn.BatchNorm2d(n_filter)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(1-self.keep_prob)
            current_dims = n_filter
        self.convs = nn.Sequential(layers)

        # FCs
        layers_fc = OrderedDict()
        current_dims = self.n_filters[-1]
        for i, n in enumerate(self.fc):
            layers_fc['fc{}'.format(i+1)] = nn.Linear(current_dims, n)
            if self.batch_norm:
                layers_fc['bn{}'.format(i + 1)] = nn.BatchNorm1d(n)
                layers_fc['relu{}'.format(i+1)] = nn.ReLU()
                layers_fc['drop{}'.format(i+1)] = nn.Dropout(1-self.keep_prob)
            current_dims = n
        self.fcs = nn.Sequential(layers_fc)

        self.classifier = nn.Sequential(nn.Linear(current_dims, self.num_classes))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        x = self.convs.forward(input)

        # global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.fcs.forward(x)

        x = self.classifier(x)

        return x























