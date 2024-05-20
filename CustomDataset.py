#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
#from skimage import io
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, CutOff = None,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.CutOff = CutOff
        
    def __len__(self):
        if self.CutOff:
            return self.CutOff
        else:
            return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        #image = io.imread(img_path).convert('RGB')
        image = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        
        
        if self.transform:
            image = self.transform(image)

        return (self.annotations.iloc[index,0], image,y_label)


