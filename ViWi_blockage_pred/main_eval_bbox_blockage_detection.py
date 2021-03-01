'''
The main script for the baseline algorithm of ViWi blockage prediction. 
------
Author: Gouranga Charan
Nov. 2021
'''

import sys 

import torch
from build_net_alternate_bbox import RecNet
from model_eval_bbox import modelEval
from data_feed_bbox import DataFeed
import torchvision.transforms as trf
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io, transform
import sklearn


print('The scikit-learn version is {}.'.format(sklearn.__version__))
#stdoutOrigin=sys.stdout 
#sys.stdout = open("log.txt", "w")

def main(): 
    
    
    # Experiment options:
    
    
    options_dict = {
        'tag': 'Exp1_beam_seq_pred_no_images',
        'operation_mode': 'beams',
    
        # Data:
        'train_ratio': 1,
        'test_ratio': 1,
        'img_mean': (0.4905,0.4938,0.5285),
        'img_std':(0.05922,0.06468,0.06174),
        'val_data_file': 'proactive_handoff_bbox_eval.csv',
        'results_file': 'blockage_prediction_results_beam_only_2layers.mat',
    
        # Net:
        'net_type':'gru',
        'cb_size': 129,  # Beam codebook size
        'out_seq': 1,  # Length of the predicted sequence
        'inp_seq': 8, # Length of inp beam and image sequence
        'embed_dim': 512,  # Dimension of the embedding space (same for images and beam indices)
        'hid_dim': 20,  # Dimension of the hidden state of the RNN
        'img_dim': [3, 416, 416],  # Dimensions of the input image
        'out_dim': 2,  # Dimensions of the softmax layers
        'num_rec_lay': 2,  # Depth of the recurrent network
        'drop_prob': 0.2,
    
        # Train param
        'gpu_idx': 0,
        'solver': 'Adam',
        'shf_per_epoch': True,
        'num_epochs': 90,
        'batch_size':1000,
        'val_batch_size':1,
        'lr': 1e-3,
        'lr_sch': [30, 60],
        'lr_drop_factor':0.1,
        'wd': 0,
        'display_freq': 5,
        'coll_cycle': 5,
        'val_freq': 20,
        'prog_plot': False,
        'fig_c': 0
    }
    
    
    # Fetch training data
    
    
    
    resize = trf.Resize((options_dict['img_dim'][1],options_dict['img_dim'][2]))
    normalize = trf.Normalize(mean=options_dict['img_mean'],
                              std=options_dict['img_std'])
    transf = trf.Compose([
        trf.ToPILImage(),
        resize,
        trf.ToTensor(),
        normalize
    ])

    
    val_feed = DataFeed(root_dir=options_dict['val_data_file'],
                         n=options_dict['inp_seq']+options_dict['out_seq'],
                         img_dim=tuple(options_dict['img_dim']),
                         transform=transf)
    val_loader = DataLoader(val_feed,batch_size=1)
    options_dict['test_size'] = val_feed.__len__()
    
    dataset_sizes = {
        
        'valid': len(val_loader.dataset)
    }
    print(dataset_sizes)
    Tensor = torch.cuda.FloatTensor
    
   
    with torch.cuda.device(options_dict['gpu_idx']):
    
        # Build net:
        # ----------
        if options_dict['net_type'] == 'gru':
            net = RecNet(options_dict['embed_dim'],
                         options_dict['hid_dim'],
                         options_dict['out_dim'],
                         options_dict['out_seq'],
                         options_dict['num_rec_lay'],
                         options_dict['drop_prob'],
                         )
            net = net.cuda()
    
        # Train and test:
        # ---------------
        net, options_dict = modelEval(net, 
                                       val_loader,
                                       options_dict)
    
def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    
if __name__ == "__main__":
    main()