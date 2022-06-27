import argparse
ap = argparse.ArgumentParser(description="test")
ap.add_argument("-n", "--name", required=True, type=str, help="New model name")
args = ap.parse_args()

model_name = args.name


import os
os.chdir("/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking")

import sys
sys.path.append('/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking')

import echonet
from echonet.datasets import Echo

import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.data import Dataset, DataLoader, Subset
from multiprocessing import cpu_count

from src.utils.torch_utils import TransformDataset, torch_collate
from src.transform_utils import generate_2dmotion_field
from src.loss_functions import huber_loss, convert_to_1hot, convert_to_1hot_tensor

from src.model.R2plus1D_18_MotionNet import R2plus1D_18_MotionNet  # ORIGINAL MODEL

# updated models
# dropout not in place, and also might have been in a weird location.
from src.model.dropout_0_10_R2plus1D_18_MotionNet import dropout_0_10_R2plus1D_18_MotionNet 
from src.model.dropout_0_25_R2plus1D_18_MotionNet import dropout_0_25_R2plus1D_18_MotionNet 
from src.model.dropout_0_50_R2plus1D_18_MotionNet import dropout_0_50_R2plus1D_18_MotionNet 
from src.model.dropout_0_75_R2plus1D_18_MotionNet import dropout_0_75_R2plus1D_18_MotionNet 
# dropout? (didn't have forward pass defined, but still saw different outputs??)
from src.model.dropout_v2_0_00_R2plus1D_18_MotionNet import dropout_v2_0_00_R2plus1D_18_MotionNet 
from src.model.dropout_v2_0_10_R2plus1D_18_MotionNet import dropout_v2_0_10_R2plus1D_18_MotionNet 
from src.model.dropout_v2_0_25_R2plus1D_18_MotionNet import dropout_v2_0_25_R2plus1D_18_MotionNet 
from src.model.dropout_v2_0_50_R2plus1D_18_MotionNet import dropout_v2_0_50_R2plus1D_18_MotionNet 
from src.model.dropout_v2_0_75_R2plus1D_18_MotionNet import dropout_v2_0_75_R2plus1D_18_MotionNet 
# dropout with what I think is properly defined behavior in the models.
from src.model.dropout_v3_0_00_R2plus1D_18_MotionNet import dropout_v3_0_00_R2plus1D_18_MotionNet 
from src.model.dropout_v3_0_10_R2plus1D_18_MotionNet import dropout_v3_0_10_R2plus1D_18_MotionNet 
from src.model.dropout_v3_0_25_R2plus1D_18_MotionNet import dropout_v3_0_25_R2plus1D_18_MotionNet 
# multiple dropout layers (4)
from src.model.dropout_v4_0_00_R2plus1D_18_MotionNet import dropout_v4_0_00_R2plus1D_18_MotionNet 
from src.model.dropout_v4_0_10_R2plus1D_18_MotionNet import dropout_v4_0_10_R2plus1D_18_MotionNet 
from src.model.dropout_v4_0_25_R2plus1D_18_MotionNet import dropout_v4_0_25_R2plus1D_18_MotionNet 



from src.echonet_dataset import EchoNetDynamicDataset
from src.clasfv_losses import deformation_motion_loss, motion_seg_loss, DiceLoss, categorical_dice
from src.train_test import train, test, train_with_log, test_with_log

import numpy as np
import matplotlib.pyplot as plt

import random
import pickle
import time

tic, toc = (time.time, time.time)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Tensor = torch.cuda.FloatTensor


# try to make reproducibility easier.
torch.manual_seed(0)
np.random.seed(0)


with open("fold_indexes/stanford_train_sampled_indices", "rb") as infile:
    train_mask = pickle.load(infile)
infile.close()

with open("fold_indexes/stanford_valid_sampled_indices", "rb") as infile:
    valid_mask = pickle.load(infile)
infile.close()


batch_size = 4
num_workers = max(4, cpu_count()//2)

def worker_init_fn_valid(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def worker_init_fn(worker_id):
    # See here: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    # and the original post of the problem: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817373837
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def permuter(list1, list2):
    for i1 in list1:
        for i2 in list2:
            yield (i1, i2)
            

param_trainLoader = {'collate_fn': torch_collate,
                     'batch_size': batch_size,
                     'num_workers': max(4, cpu_count()//2),
                     'worker_init_fn': worker_init_fn}

param_testLoader = {'collate_fn': torch_collate,
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': max(4, cpu_count()//2),
                    'worker_init_fn': worker_init_fn}

paramLoader = {'train': param_trainLoader,
               'valid': param_testLoader,
               'test':  param_testLoader}


train_dataset = EchoNetDynamicDataset(split='train', subset_indices=train_mask, period=1)
valid_dataset = EchoNetDynamicDataset(split='val', subset_indices=valid_mask, period=1)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, 
                              shuffle=True, pin_memory=("cuda"), 
                              worker_init_fn=worker_init_fn,
                              drop_last=True)

valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, 
                              num_workers=num_workers,
                              shuffle=False, pin_memory=("cuda"),
                              worker_init_fn=worker_init_fn_valid
                             )



if model_name == "dropout_v3_0_00_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v3_0_00_R2plus1D_18_MotionNet()
        
elif model_name == "dropout_v3_0_10_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v3_0_10_R2plus1D_18_MotionNet()
    
elif model_name == "dropout_v3_0_25_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v3_0_25_R2plus1D_18_MotionNet()
    
elif model_name == "dropout_v4_0_00_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v4_0_00_R2plus1D_18_MotionNet()
    
elif model_name == "dropout_v4_0_10_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v4_0_10_R2plus1D_18_MotionNet()
    
elif model_name == "dropout_v4_0_25_R2plus1D_18_MotionNet.pth":
    model_template_obj = dropout_v4_0_25_R2plus1D_18_MotionNet()


model = torch.nn.DataParallel(model_template_obj)
model.to("cuda")

print(f'{model_name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')

lr_T = 1e-4 
optimizer = optim.Adam(model.parameters(), lr=lr_T)


model_save_path = f"tmp_save_models/{model_name}"

train_loss_list = []
valid_loss_list = []
total_train_time = 0

n_epoch = 10
min_loss = 1e5

with open(f"./tmp_save_models/{model_name}_training_log.txt", "a") as log_file:
    for epoch in range(1, n_epoch + 1):
        log_file.write(f"{'-' * 32} Epoch {epoch} {'-' * 32}\n")
        start = time.time()
        train_loss = train_with_log(epoch, train_loader=train_dataloader, model=model, optimizer=optimizer, log_file = log_file)
        train_loss_list.append(np.mean(train_loss))
        end = time.time()
        log_file.write("training took {:.8f} seconds\n".format(end-start))
        total_train_time += (end - start)
        valid_loss = test_with_log(epoch, test_loader=valid_dataloader, model=model, optimizer=optimizer, log_file = log_file)
        valid_loss_list.append(np.mean(valid_loss))

        if (np.mean(valid_loss) < min_loss) and (epoch > 0):
            min_loss = np.mean(valid_loss)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_save_path)

        if epoch == 3:
            lr_T = 1e-5
            optimizer = optim.Adam(model.parameters(), lr=lr_T)
            
        # try to get rid of runtime cuda errors (illegal mem access)
        torch.cuda.empty_cache()

    log_file.write(f'total training took: {total_train_time // 60} m {total_train_time % 60:.2f}s\n')