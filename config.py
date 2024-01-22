import torch
import os


class Config(object):
    noise_dim = 256
    batch_size = 40
    num_works = 4
    # Epoch: 迭代次数
    epochs = 500
    # 网络隐藏特征图尺寸 (判别器的复杂度要略低于生成器,避免过度指导图片生成)
    gen_feature_map = 256
    disc_feature_map = 128
    # 学习率
    gen_lr = 2e-4
    disc_lr = 1e-4
    # 保存形式
    modelSave = False
    autoLoad = True
    # 存储路径
    data_path = "dataset"
    create_model_path = "" # "checkpoints/2024-0117-1817/gen3_params_7.4841.pt"
    pre_gen_path = "" # "checkpoints/2024-0118-1138/last_gen_params.pt"
    pre_disc_path = "" # "checkpoints/2024-0118-1138/last_disc_params.pt"
    # 运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def GetLastParams():
    path_list = os.listdir("checkpoints")
    if len(path_list) > 1:
        target_path = "checkpoints/" + sorted(path_list)[-2]
    else:
        print("No checkpoints found!")
        target_path = ""
    return target_path
