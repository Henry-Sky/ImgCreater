import torch

class Config(object):
    image_dim = 96 * 96 * 3
    noise_dim = 100
    batch_size = 40
    num_works = 4
    # Epoch超过select_epoch后开始保存损失较低的中间模型
    select_epoch = 100
    epochs = 1000
    # 网络隐藏特征图尺寸 (判别器的复杂度要略低于生成器,避免过度指导图片生成)
    gen_feature_map = 128
    disc_feature_map = 64
    # 学习率
    gen_lr = 3e-4
    disc_lr = 1e-4
    # 存储路径
    data_path = "dataset"
    create_model_path = "checkpoints/2024-0117-1817/gen3_params_7.4841.pt" # "your img create path"
    pre_gen_path = "checkpoints/2024-0118-0838/last_gen_params.pt" # "your pre trained checkpoints"
    pre_disc_path = "checkpoints/2024-0118-0838/last_disc_params.pt" # "your pre trained checkpoints"
    # 运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")