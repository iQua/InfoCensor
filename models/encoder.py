import torch
from torch import nn
from torch.nn import functional as F

class mlp_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None, bn=True, drop_rate=0.0, info_model=False):
        super(mlp_encoder, self).__init__()
        self.info_model = info_model
        n_h = 128
        modules = []
        if hidden_dims is None:
            hidden_dims = [n_h*2, n_h]

        hidden_dims = [in_dim] + hidden_dims

        # Build Encoder
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

        self.pre_encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1], out_dim)
        self.logvar = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, input):
        encoding = self.pre_encoder(input)
        mu = self.mu(encoding)

        if not self.info_model:
            return mu
        else:
            logvar = self.logvar(encoding)
            sigma = torch.exp(0.5*logvar)
            if self.training:
                return mu, sigma
            else:
                return mu + sigma*torch.randn_like(sigma)


class mlp_encoder_for_mi_calculation(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None, bn=True, drop_rate=0.0, info_model=True):
        super(mlp_encoder_for_mi_calculation, self).__init__()
        self.info_model = info_model
        n_h = 128
        modules = []
        if hidden_dims is None:
            hidden_dims = [n_h*2, n_h]

        hidden_dims = [in_dim] + hidden_dims

        # Build Encoder
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

        self.pre_encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dims[-1], out_dim)
        self.logvar = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, input):

        encoding = self.pre_encoder(input)
        mu = self.mu(encoding)
        logvar = self.logvar(encoding)
        sigma = torch.exp(0.5*logvar)

        return mu, sigma

class lenet_encoder_for_mi_calculation(nn.Module):
    def __init__(self, img_dim, out_dim, drop_rate=0.0, info_model=True):
        super(lenet_encoder_for_mi_calculation, self).__init__()
        self.drop_rate = drop_rate
        self.info_model = info_model

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        f_dim = (((img_dim-2)//2 - 2)//2-2)//2
        self.fc1 = nn.Linear(f_dim*f_dim*64, 128)
        self.mu = nn.Linear(128, out_dim)
        self.logvar = nn.Linear(128, out_dim)

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

        mu = self.mu(out)
        logvar = self.logvar(out)
        sigma = torch.exp(0.5*logvar)

        return mu, sigma


class lenet_encoder(nn.Module):
    def __init__(self, img_dim, out_dim, drop_rate=0.0, info_model=False):
        super(lenet_encoder, self).__init__()
        self.drop_rate = drop_rate
        self.info_model = info_model

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        f_dim = (((img_dim-2)//2 - 2)//2-2)//2
        self.fc1 = nn.Linear(f_dim*f_dim*64, 128)
        self.mu = nn.Linear(128, out_dim)
        self.logvar = nn.Linear(128, out_dim)

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

        mu = self.mu(out)

        if not self.info_model:
            return mu
        else:
            logvar = self.logvar(out)
            sigma = torch.exp(0.5*logvar)
            if self.training:
                return mu, sigma
            else:
                return mu + sigma*torch.randn_like(sigma)


class lstm_encoder(nn.Module):
    def __init__(self, voc_size, embedding_dim, out_dim, hidden_dim=None, drop_rate=0.2, info_model=False):

        super(lstm_encoder, self).__init__()

        if hidden_dim is None: hidden_dim = out_dim*2
        self.drop_rate = drop_rate
        self.info_model = info_model

        self.embedding = nn.Embedding(voc_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim*2)

        self.fc = nn.Linear(hidden_dim*2, hidden_dim)

        self.mu = nn.Linear(hidden_dim, out_dim)
        self.logvar = nn.Linear(hidden_dim, out_dim)


    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        #embedded = [sent len, batch size, emb dim]

        output, (hn, cn) = self.lstm(embedded)

        #output = [sent len, batch size, hid dim]
        #hn = [1, batch size, hid dim]
        try:
            assert torch.equal(output[-1,:,:], hn.squeeze(0))
        except:
            print(output[-1,:,:], hn.squeeze(0))
            exit()

        hn = F.dropout(hn.squeeze(0), p=self.drop_rate, training=self.training)
        encoding = F.dropout(self.fc(hn), p=self.drop_rate, training=self.training)

        mu = self.mu(encoding)

        if not self.info_model:
            return mu
        else:
            logvar = self.logvar(encoding)
            sigma = torch.exp(0.5*logvar)
            if self.training:
                return mu, sigma
            else:
                return mu + sigma*torch.randn_like(sigma)


class vgg_extractor(nn.Module):
    def __init__(self, info_model=False):
        super(vgg_extractor, self).__init__()

        self.info_model = info_model

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.mu = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        if info_model:
            self.logvar = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    def forward(self, x):
        encoding = self.convnet(x)

        mu = self.mu(encoding)

        if not self.info_model:
            return mu
        else:
            logvar = self.logvar(encoding)
            sigma = torch.exp(0.5*logvar)
            if self.training:
                return mu, sigma
            else:
                return mu + sigma*torch.randn_like(sigma)
