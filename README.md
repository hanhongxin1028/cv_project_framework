# 1. 介绍
DeepLearning图像处理方向项目的大体框架。

方便你专注于核心代码的编写, 并且让项目的结构更加清晰



# 2. 如何使用？
将仓库克隆到你的本地，可以使用以下命令：
```bash
git clone https://github.com/hanhongxin1028/cv_project_framework.git

# 3. 文件介绍
* `data`下存放数据集
* `models`下可存放你的 损失函数 和 网络结构
* `src`下存放核心代码, 具体文件介绍如下

| 文件            | 存放                                          |
| --------------- | --------------------------------------------- |
| data_augment.py | 数据增强 代码                                 |
| data_loader.py  | 数据加载 代码                                 |
| main.py         | 程序 主入口, 并放置了超参数、模型、数据加载等 |
| test.py         | 测试 代码                                     |
| train.py        | 训练、验证、测试 代码                         |
| utils.py        | 工具类函数 代码                               |

* `.gitignore`中一般将data和weights添加上，因为他们并不是代码无需用git管理
