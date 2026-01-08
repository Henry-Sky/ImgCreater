import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# 任务: 生成手写数据集

# 判别器
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        # 模型层次结构
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            # LeakyReLU 相比 ReLU 可以解决死亡神经元问题
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# 生成器
class Generator(nn.Module):
    # 噪声维度, 生成图片维度
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# 超参数(Hyperparameters)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 3e-4
z_dim = 64  # 噪声的维度
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

# 实例化判别器
disc = Discriminator(image_dim).to(device)
# 实例化生成器
gen = Generator(z_dim, image_dim).to(device)

# 噪声
fixed_noise = torch.randn(batch_size, z_dim).to(device)
# 数据变换器
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])
# 数据集
dataset = datasets.MNIST(root="./data/", transform=transforms, download=True)
# 加载器
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# 优化器
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
# 损失函数
criterion = nn.BCELoss()
# 运行日志输出(生成器生成的数据)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        # 随机噪声(生成器的输入数据)
        noise = torch.randn(batch_size, z_dim).to(device)
        # 生成图
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))  # 真图损失
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # 假图损失
        # 判别器判别真假的损失
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator min log(1 - D(G(z))) <-> max log(D(G(z)))

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
