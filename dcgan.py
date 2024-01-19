import torch.nn as nn

class NetG(nn.Module):
    def __init__(self, config):
        super(NetG, self).__init__()
        # 噪声维度
        noise_dim = config.noise_dim
        # feature_dim: 隐藏特征尺寸
        feature_dim = config.gen_feature_map

        self.gen = nn.Sequential(
            # 对输入的的噪声进行转置卷积
            nn.ConvTranspose2d(noise_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            # 输出: size: 4*4`dim: feature_dim * 8

            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            # 输出: size: 8*8 dim: feature_dim * 4

            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            # 输出: size: 16 x 16 dim: feature_dim * 2

            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            # 输出: size: 32 x 32 dim: feature_dim

            nn.ConvTranspose2d(feature_dim, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # 输出: size: 96 x 96 dim: feature_dim
        )
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for layer in self.gen:
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data.normal_(0.01, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, input):
        return self.gen(input)


class NetD(nn.Module):
    def __init__(self, config):
        super(NetD, self).__init__()
        feature_dim = config.disc_feature_map

        self.disc = nn.Sequential(
            # 输入: size: 96 x 96 dim: 3
            nn.Conv2d(3, feature_dim, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: size: 32 x 32`dim: feature_dim

            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: size: 16 x 16`dim: feature_dim * 2

            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: size: 8 x 8`dim: feature_dim * 4

            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: size: 4 x 4`dim: feature_dim * 8

            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: size: 1 x 1`dim: 1
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.disc:
            if isinstance(layer,nn.Conv2d):
                layer.weight.data.normal_(0.0, 0.02)
            if isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, input):
        return self.disc(input).view(-1)
