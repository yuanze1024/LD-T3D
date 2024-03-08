import os
import torch
import numpy as np
from PIL import Image
import pandas as pd

def get_dataset(modality, source_id_list, data_path, angle=None):
    if modality == "3D":
        return PointCloudDataset(source_id_list, data_path)
    elif modality == "image":
        return ImageDataset(source_id_list, data_path, angle)
    elif modality == "text":
        return TextDataset(source_id_list, data_path)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, source_id_list, data_path):
        self.id_list = source_id_list
        self.data_path = data_path
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, index):
        id = self.id_list[index]
        data = np.load(self.data_path + '/' + id + '.npy')
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return data
    
    def __len__(self):
        return len(self.id_list)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_id_list, data_path, angle):
        self.id_list = source_id_list
        self.data_path = data_path
        self.angle = angle
        
    def __getitem__(self, index):
        id = self.id_list[index]
        return Image.open(os.path.join(self.data_path, id, id + f'_00{self.angle:02d}.webp')).convert('RGB')
    
    def __len__(self):
        return len(self.id_list)
    

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, source_id_list, data_path):
        self.id_list = source_id_list
        self.caption_df = pd.read_csv(data_path, header=None)
        self.caption_df.columns = ['id', 'caption']
        self.caption_df = self.caption_df.set_index('id')
        
    def __getitem__(self, index):
        id = self.id_list[index]
        return self.caption_df.at[id, 'caption'].strip().strip('"').capitalize()
    
    def __len__(self):
        return len(self.id_list)