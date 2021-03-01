'''
Data feeding class. It generates a list of data samples, each of which is a python list of
tuples. Every tuple consists of an image path and a beam index. Since this class is used in
the baseline solution, it only outputs sequences of beam indices, and it ignores the images.
-------------------------------
Author: Gouranga Charan
Nov. 2021
'''

import os
import numpy as np
import pandas as pd
import torch
import random
from skimage import io
from torch.utils.data import Dataset
import ast

############### Create data sample list #################
def create_samples(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    for idx, row in f.iterrows():
        bbox = row.values[23:31]
        data_samples.append(bbox)
    
    for idx, row in f.iterrows():
        beam_blockage = row.values[31:44].astype(np.float32)
        pred_beam_block = list(beam_blockage)   
        pred_val.append(pred_beam_block)
    
    
    print('list is ready')
    return data_samples, pred_val


#########################################################

class DataFeed(Dataset):
    """
    A class fetching a PyTorch tensor of beam indices.
    """
    
    def __init__(self, root_dir,
    			n,
    			img_dim,
    			transform=None,
    			init_shuflle=False):
    
    	self.root = root_dir
    	self.samples, self.pred_val = create_samples(self.root, shuffle=False)
    	self.transform = transform
    	self.seq_len = n
    	self.img_dim = img_dim
    
    def __len__(self):
    	return len(self.samples)
    
    def __getitem__(self, idx):
        bbox = self.samples[idx] # Read one data sample
        block_val = self.pred_val[idx]
        assert len(block_val) >= self.seq_len, 'Unsupported sequence length'   
        bbox = bbox[:self.seq_len] # Read a sequence of tuples from a sample
        block_val = block_val[:self.seq_len] # Read a sequence of tuples from a sample
        beams = torch.zeros((self.seq_len,))
        bbox_val = torch.zeros((self.seq_len,256))        
        images = []
        
        for i,s in enumerate(bbox):
            box = ast.literal_eval(s) # Read only beams
            bbox_val[i] = torch.tensor(box, requires_grad=False) 
         
        
        for i,s in enumerate( block_val ):
            x = s # Read only beams
            beams[i] = torch.tensor(x, requires_grad=False)      
        
        
        return (bbox_val,beams)