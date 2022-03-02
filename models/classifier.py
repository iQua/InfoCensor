import torch
from torch import nn
from torch.nn import functional as F

class mlp_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0, num_classes=2):
        super(mlp_classifier, self).__init__()

        self.drop_rate = drop_rate
        modules = []
        if hidden_dims is None:
            hidden_dims = []

        hidden_dims = [in_dim] + hidden_dims

        for layer_idx in range(len(hidden_dims)-1):
            if bn:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.BatchNorm1d(hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )



        self.features = None if len(modules) == 0 else nn.Sequential(*modules)
        self.logits = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.logits(features)


class binary_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(binary_classifier, self).__init__()

        self.drop_rate = drop_rate
        modules = []
        if hidden_dims is None:
            hidden_dims = []

        hidden_dims = [in_dim] + hidden_dims

        for layer_idx in range(len(hidden_dims)-1):
            if bn:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.BatchNorm1d(hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx+1]),
                        nn.ReLU(),
                        nn.Dropout(drop_rate))
                )



        self.features = None if len(modules) == 0 else nn.Sequential(*modules)
        self.logit = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.sigmoid(self.logit(features))




class vgg_classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg_classifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fcnet = nn.Sequential(
            nn.Linear(512 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(out.size(0), -1)
        out = self.fcnet(out)
        return out
