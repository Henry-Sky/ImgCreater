import os
import torch
import torchvision
from datetime import datetime
from dcgan import NetG, NetD
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from config import Config
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCELoss


def train():
    # 加载参数配置
    cfg = Config()
    device = cfg.device
    # 数据加载配置
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(cfg.data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_works,
        drop_last=True
    )

    # 实例化网络
    gen = NetG(cfg).to(device)
    disc = NetD(cfg).to(device)
    # 加载预训练参数(如果存在)
    if os.path.exists(cfg.pre_gen_path):
        gen.load_state_dict(torch.load(cfg.pre_gen_path))
        print("生成器预训练参数加载成功!")
    if os.path.exists(cfg.pre_disc_path):
        disc.load_state_dict(torch.load(cfg.pre_disc_path))
        print("判别器预训练参数加载成功!")

    # 定义优化器
    opt_gen = Adam(gen.parameters(), cfg.gen_lr)
    opt_disc = Adam(disc.parameters(), cfg.disc_lr)
    criterion = BCELoss().to(device)

    # 创建运行数据文件夹
    start_time = datetime.now().strftime("%Y-%m%d-%H%M")
    if not os.path.exists(f"checkpoints/{start_time}"):
        os.mkdir(f"checkpoints/{start_time}")
    # 运行日志输出(生成器生成的数据)
    fixed_noise = torch.randn(cfg.batch_size, cfg.noise_dim, 1, 1).to(device)
    writer_fake = SummaryWriter(f"runs/Animation/{start_time}/fake")
    writer_real = SummaryWriter(f"runs/Animation/{start_time}/real")
    step = 0

    # 模型训练
    min_gloss = 100
    # 模型迭代
    for epoch in range(5):
        # 模型指标
        lossD_all = 0
        lossG_all = 0
        batch_num = 0
        # 批次迭代
        for batch_idx, (real_img, _) in enumerate(dataloader):
            batch_num += 1
            real = real_img.to(device)
            batch_size = real.shape[0]

            """
            训练判别器
            step 1 : 生成器生成一张假图
            step 2 : 从数据集中调取一张真图
            step 3 : 计算真图, 假图损失, 判别器损失
            """
            # 生成器基于随机噪声"生成器"生成一组假图
            noise = torch.randn(batch_size, cfg.noise_dim, 1, 1).to(device)
            fake = gen.forward(noise)
            # 计算损失
            disc_real_labs = disc.forward(real).view(-1)
            disc_fake_labs = disc.forward(fake).view(-1)
            lossD_real = criterion(disc_real_labs, torch.ones_like(disc_real_labs))
            lossD_fake = criterion(disc_fake_labs, torch.zeros_like(disc_fake_labs))
            lossD = (lossD_real + lossD_fake) / 2
            lossD_all += lossD.item()
            # 计算梯度并传递
            disc.zero_grad()  # 消除上一次的梯度影响
            lossD.backward(retain_graph=True)
            opt_disc.step()

            """
            训练生成器
            step 1 : 对生成器生成的假图判断
            step 2 : 计算判断器对假图的损失
            step 3 : 更新参数
            """
            output = disc.forward(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            lossG_all += lossG.item()
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            """
            tensorboard 可视化训练过程
            """
            if batch_idx == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 3, 96, 96)
                    data = real.reshape(-1, 3, 96, 96)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Fake Images", img_grid_fake, global_step=step
                    )

                    writer_real.add_image(
                        "Real Images", img_grid_real, global_step=step
                    )

                    step += 1
        # 更新一次迭代的平均损失
        lossD_all = lossD_all / batch_num
        lossG_all = lossG_all / batch_num

        """
        保存模型
        """
        if epoch > cfg.select_epoch:

            if lossG_all < min_gloss:
                min_gloss = lossG_all
                torch.save(gen.state_dict(), f"checkpoints/{start_time}/gen_{epoch + 1}_params_({lossG_all:.4f}).pt")
                torch.save(disc.state_dict(), f"checkpoints/{start_time}/disc_{epoch + 1}_params_({lossD_all:.4f}).pt")
                print("min loss checkpoints saved")

        torch.save(gen.state_dict(), f"checkpoints/{start_time}/last_gen_params.pt")
        torch.save(disc.state_dict(), f"checkpoints/{start_time}/last_disc_params.pt")

        print(f"Epoch:{epoch + 1}, LossD:{lossD_all:.4f}, LossG:{lossG_all:.4f}")


if __name__ == '__main__':
    train()
