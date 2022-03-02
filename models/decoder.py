import torch
from torch import nn
from torch.nn import functional as F

class mlp_decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None, bn=True, drop_rate=0.0):
        super(mlp_decoder, self).__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        hidden_dims = [in_dim] + hidden_dims

        # Build Decoder
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

        self.pre_decoder = nn.Sequential(*modules)
        ## since all the data is normalized into [0, 1]
        self.decoding = nn.Sequential(nn.Linear(hidden_dims[-1], out_dim), nn.Sigmoid())

    def forward(self, input):
        return self.decoding(self.pre_decoder(input))



class lenet_decoder(nn.Module):
    def __init__(self, input_dim, img_dim, img_channels=3, drop_rate=0.0):
        super(lenet_decoder, self).__init__()
        self.drop_rate = drop_rate

        ## the decoder's hyperparameters correspond to the encoder's hyperparameters
        kernel_sizes = [3, 3, 3]
        stride = 1
        upsample_sizes = [((img_dim-2)//2 - 2)//2, (img_dim-2)//2, img_dim-2]
        self.init_feat_dim = (((img_dim-2)//2 - 2)//2-2)//2

        self.fc = nn.Linear(input_dim, 64*self.init_feat_dim*self.init_feat_dim)


        self.upsample1 = nn.Upsample(size=upsample_sizes[0], mode='bicubic')
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_sizes[0], stride=stride)
        self.upsample2 = nn.Upsample(size=upsample_sizes[1], mode='bicubic')
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=kernel_sizes[1], stride=stride)
        self.upsample3 = nn.Upsample(size=upsample_sizes[2], mode='bicubic')
        self.decoding = nn.Sequential(nn.ConvTranspose2d(16, img_channels, kernel_size=kernel_sizes[2], stride=stride),
                                      nn.Sigmoid())


    def forward(self, x):
        out = self.fc(x)
        out = F.relu(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = out.view(out.size(0), 64, self.init_feat_dim, self.init_feat_dim)
        out = self.upsample1(out)
        out = F.relu(self.deconv1(out))
        out = self.upsample2(out)
        out = F.relu(self.deconv2(out))
        out = self.upsample3(out)

        return self.decoding(out)




# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes: int = 1000) -> None:
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

### decoder
class lstm_decoder(nn.Module):
    def __init__(self, voc_size, fea_dim, hidden_dim, drop_rate=0.2):

        super(lstm_decoder, self).__init__()

        self.drop_rate = drop_rate
        self.voc_size = voc_size

        self.lstm = nn.LSTM(fea_dim, hidden_dim) ## [batch, fea_dim] dim of the features

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, features, sen_length=1):

        output = []
        for idx in range(sen_length):
            if idx == 0:
                ### features.unsqueeze(0) [1, batch, fea_dim]
                out_idx, (h_idx, c_idx) = self.lstm(features.unsqueeze(0)) ## [1, batch, hidden]
            else:
                out_idx, (h_idx, c_idx) = self.lstm(features.unsqueeze(0), (h_idx, c_idx))

            out_idx = self.fc2(self.fc1(out_idx))
            output.append(self.sigmoid(out_idx)*(self.voc_size-1))

        output = torch.cat(output, dim=0).squeeze()

        return output


class vgg_decoder(nn.Module):
    def __init__(self, img_dim=50, img_channels=3):
        super(vgg_decoder, self).__init__()

        ## the decoder's hyperparameters correspond to the encoder's hyperparameters
        kernel_sizes = [3, 3, 3]
        stride = 1
        upsample_sizes = [(img_dim-2)//2, img_dim-2]

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_sizes[0], stride=stride)
        self.upsample1 = nn.Upsample(size=upsample_sizes[0], mode='bicubic')
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_sizes[1], stride=stride)
        self.upsample2 = nn.Upsample(size=upsample_sizes[1], mode='bicubic')
        self.decoding = nn.Sequential(nn.ConvTranspose2d(32, img_channels, kernel_size=kernel_sizes[2], stride=stride),
                                      nn.Sigmoid())


    def forward(self, x):

        out = F.relu(self.deconv1(x))
        out = self.upsample1(out)
        out = F.relu(self.deconv2(out))
        out = self.upsample2(out)

        return self.decoding(out)
