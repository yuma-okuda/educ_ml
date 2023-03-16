import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# class MyDataset(Dataset): 
#     def __init__(self, paths, labels):
#         super().__init__()
#         self.pathList = paths
#         self.labelList = labels
#         self.transforms = VT.Compose([
#             # VT.ToTensor(),
#             AT.MelSpectrogram(
#                 sample_rate=96000,
#                 n_fft=512,
#                 hop_length=128,
#                 n_mels=128,
#             )
#         ])
#     def __len__(self):
#         return len(self.pathList)
#     def __getitem__(self, idx):
#         file_name = self.pathList[idx]
#         label = self.labelList[idx]
#         data, _ = torchaudio.load(file_name)
#         data = data[0]
#         melSpectrogram = self.transforms(data)
#         res = melSpectrogram.flatten()
#         return res, label

class MyDataset(Dataset):
  def __init__(self,img_list):
    self.img_list = img_list
    self.prepocess = T.Compose([T.Resize((128,128)),
                                T.ToTensor(),
                                ])
  def __len__(self):
    return len(self.img_list)

  def __getitem__(self,idx):
    img = Image.open(self.img_list[idx])
    img = self.prepocess(img)
    return img
