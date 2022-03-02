import torch
from torch import nn
from torch.nn import functional as F

class mlp_discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(mlp_discriminator, self).__init__()

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
        self.logits = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        features = F.dropout(input, p=self.drop_rate, training=self.training)
        if self.features is not None: features = self.features(features)
        return self.sigmoid(self.logits(features))




class lenet_discriminator(nn.Module):
    def __init__(self, img_dim, out_dim, drop_rate=0.0, info_model=False):
        super(lenet_discriminator, self).__init__()
        self.drop_rate = drop_rate
        self.info_model = info_model

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        f_dim = (((img_dim-2)//2 - 2)//2-2)//2
        self.fc1 = nn.Linear(f_dim*f_dim*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.fc2(out)

        return self.sigmoid(out)
