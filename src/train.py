import torch
from tqdm import tqdm


def train(model, your_loss, train_loader, epoch, optimizer, device, scaler):
    """
        作用 :
            训练模型
        参数 : 
             model : 定义的模型
             your_loss : 损失函数
             train_loader : 训练数据集
             epoch : 迭代数
             optimizer : 优化器
             device : 设备
             scaler : 混合精度
    """
    # 1. 初始化变量
    epoch_loss = 0.0  # 存储每个epoch的总损失合
    num_batches = len(train_loader)  # 记录总批次数
    pbar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True)  # 进度条
    
    
    # 2. 遍历数据集，进行训练
    for batch_id, () in enumerate(train_loader):
        # 2.1 数据移动GPU设备

        # 2.2 前向传播
        model()

        # 2.3 compute loss
        loss = your_loss()

        # 2.4 反向传播 (混合精度)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # 2.5 累加整轮的损失和准确率
        epoch_loss += loss.item()

        # 2.6 更新 tqdm 进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
        pbar.update(1)  # 更新进度条

    pbar.close()    

    # 3. 每个epoch结束后，计算平均损失(采用一种方式记录下来每一轮的平均损失)
    avg_loss = epoch_loss / num_batches
            
    # 4. 每一个epoch清理一次 GPU 缓存，释放未使用的内存
    torch.cuda.empty_cache() 

def valid(model, your_loss, val_loader, epoch, device):
    """
        作用 :
            验证模型
        参数 : 
             model : 定义的模型
             valid_loader : 验证数据集
             epoch : 迭代数
             log_utils : 记录日志
             device : 设备
    """
    # 1. 初始化变量
    epoch_loss = 0.0  # 存储每个epoch的总损失
    num_batches = len(val_loader)  # 记录总批次数
    pbar = tqdm(val_loader, total=num_batches, desc=f"Valid Epoch {epoch}", dynamic_ncols=True)  # 进度条

    # 2. 遍历数据集 , 进行验证
    with torch.no_grad():
        for batch_id, () in enumerate(val_loader):
            # 2.1 数据移动GPU设备

            # 2.2 前向传播
            model()

            # 2.3 compute loss
            loss = your_loss()

            

            # 2.4 累加整轮的损失
            epoch_loss += loss.item()


            # 2.7 进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
            
            pbar.update(1)  # 更新进度条

    pbar.close()

    # 3. 每个epoch结束后，计算平均损失(注意记录)
    avg_loss = epoch_loss / num_batches
    
    # 每一个epoch清理一次 GPU 缓存，释放未使用的内存
    torch.cuda.empty_cache() 
    

def test(model, test_loader, device):
    pass