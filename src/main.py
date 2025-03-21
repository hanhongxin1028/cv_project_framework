import torch
from torch.utils.data import DataLoader
from data_loader import YourDataset   # 加载数据集
from models.your_model import YourModel  # 加载模型
from models.your_loss import YourLoss  # 加载损失函数
from train import train, valid, test  # 加载训练、验证、测试函数

'''
    一、定义超参数、模型、损失函数、优化器
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 0
batch_size = 0
learning_rate = 0

model = YourModel().to(device)
your_loss = YourLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scaler = torch.GradScaler(enabled=True)


'''
    二、定义数据集
'''
# 数据集文件路径
Image_dir = ''


# 数据增强 
train_dataset = YourDataset(Image_dir)
val_dataset = YourDataset(Image_dir)
test_dataset = YourDataset(Image_dir)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=30, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=30, pin_memory = True)



'''
    三、训练
'''
for epoch in range(1,epochs+1):
    # 训练
    model.train()
    train(model, your_loss, train_loader, epoch, optimizer, device, scaler)

    if epoch % 10 == 0:
        # 验证
        model.eval() 
        valid(model, your_loss, val_loader, epoch, device)
        # 保存模型权重
        torch.save(model.state_dict(), "weights/weight.pth")
        

# 测试
model.eval()
test(model, test_loader, device)
    