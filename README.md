# ImgCreater
用DCGAN生成对抗网络生成二次元动漫头像

---
模型结构: dcgan.py  
超参数: config.py  
训练模型: train.py  
调用模型: create.py  
---
checkpoints: 模型运行时临时保存模型参数  
dataset: 训练用数据集(93 * 93 * 3)  
imgs: 生成图片目录  
mmodels: 保存模型目录  
runs: 运行时存"tensorboard"数据  
---
依赖:  
torch  
torchvision  
tensorboard  
tensorboardX  
