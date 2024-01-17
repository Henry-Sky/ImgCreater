import os
import torch
import torchvision
from datetime import datetime
from dcgan import NetG, NetD
from config import Config
from torchvision.utils import save_image

def create():
    cfg = Config()
    device = cfg.device
    time = datetime.now().strftime("%Y-%m%d-%H%M")
    gen = NetG(cfg).to(device)
    with torch.no_grad():
        gen.load_state_dict(torch.load(cfg.create_model_path))
        fixed_noise = torch.randn(cfg.batch_size, cfg.noise_dim, 1, 1).to(device)
        img = gen.forward(fixed_noise)
        save_image(img, f'imgs/{time}.png')

if __name__ == '__main__':
    create()


