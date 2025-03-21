import torch.nn as nn   # 导入 torch 的神经网络模块

class YourModel(nn.Module):
    def __init__(self, args):
        super(YourModel, self).__init__()

        # 定义你要使用的网络结构(按自己的需求修改)
        self.network1 = nn.linear(256, 128)
        self.network2 = nn.linear(128, 64)
        
        

    def forward(self, x):
        # 调用上述定义好的网络结构, 进行前向传播(按自己的需求修改)
        x = self.network1(x)
        x = self.network2(x)

        return x

