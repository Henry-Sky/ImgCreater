import os
import torch
from datetime import datetime
from dcgan import NetG, NetD
from config import Config, GetLastParams
from torchvision.utils import save_image


def create(config):
    device = config.device
    time = datetime.now().strftime("%Y-%m%d-%H%M")
    gen = NetG(config).to(device)
    with torch.no_grad():
        if os.path.exists(config.create_model_path):
            gen.load_state_dict(torch.load(config.create_model_path))
        else:
            gen.load_state_dict(torch.load(GetLastParams() + "/last_gen_params.pt"))
            print("未找到指定参数路径,自动加载最新检查点参数")

        fixed_noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)
        img = gen.forward(fixed_noise)
        save_image(img, f'imgs/{time}.png')
        print(f'图片保存路径: imgs/{time}.png')


if __name__ == '__main__':
    cfg = Config()
    create(config=cfg)
