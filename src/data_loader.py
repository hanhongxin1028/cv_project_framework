"""
    写下数据读取的代码
"""
from torch.utils.data import Dataset


class YourDataset(Dataset):
    def __init__(self):

        pass

        
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
       

        return 
