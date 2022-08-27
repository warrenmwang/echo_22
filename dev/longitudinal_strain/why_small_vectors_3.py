# model_name = "Original_Pretrained_R2plus1DMotionSegNet.pth"

# model_name = "dropout_v2_0_25_R2plus1DMotionSegNet.pth"
model_name = "dropout_v3_0_10_R2plus1DMotionSegNet.pth"

#####################
# import os
# os.chdir("../..")
# print(os.getcwd())
import os
os.chdir("/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking")
print(os.getcwd())
import sys
sys.path.append('/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking')




# %config Completer.use_jedi = False

import SimpleITK as itk
from LabelFusion.wrapper import fuse_images

import echonet
from echonet.datasets import Echo

import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torch.utils.data import Dataset, DataLoader, Subset
from multiprocessing import cpu_count

from src.utils.torch_utils import TransformDataset, torch_collate
from src.utils.echo_utils import get2dPucks
from src.utils.camus_validate import cleanupSegmentation
from src.transform_utils import generate_2dmotion_field
from src.visualization_utils import categorical_dice
from src.loss_functions import huber_loss, convert_to_1hot, convert_to_1hot_tensor
from src.echonet_dataset import EDESpairs, EchoNetDynamicDataset

from src.model.R2plus1D_18_MotionNet import R2plus1D_18_MotionNet

# v2 dropout (in place before motion heads, forgot to define in forward pass function, but still saw diff, weird.)
from src.model.dropout_v2_0_00_R2plus1D_18_MotionNet import dropout_v2_0_00_R2plus1D_18_MotionNet
from src.model.dropout_v2_0_10_R2plus1D_18_MotionNet import dropout_v2_0_10_R2plus1D_18_MotionNet
from src.model.dropout_v2_0_25_R2plus1D_18_MotionNet import dropout_v2_0_25_R2plus1D_18_MotionNet
from src.model.dropout_v2_0_50_R2plus1D_18_MotionNet import dropout_v2_0_50_R2plus1D_18_MotionNet
from src.model.dropout_v2_0_75_R2plus1D_18_MotionNet import dropout_v2_0_75_R2plus1D_18_MotionNet
# v3 dropout (one dropout layer defined in forward pass func, this should've been the correct way to do it.)
from src.model.dropout_v3_0_00_R2plus1D_18_MotionNet import dropout_v3_0_00_R2plus1D_18_MotionNet
from src.model.dropout_v3_0_10_R2plus1D_18_MotionNet import dropout_v3_0_10_R2plus1D_18_MotionNet
from src.model.dropout_v3_0_25_R2plus1D_18_MotionNet import dropout_v3_0_25_R2plus1D_18_MotionNet
from src.model.dropout_v3_0_50_R2plus1D_18_MotionNet import dropout_v3_0_50_R2plus1D_18_MotionNet
from src.model.dropout_v3_0_75_R2plus1D_18_MotionNet import dropout_v3_0_75_R2plus1D_18_MotionNet
# v4 dropout (4 dropout layers in different places in the forward func, I'm going to guess more "generalizable")
from src.model.dropout_v4_0_00_R2plus1D_18_MotionNet import dropout_v4_0_00_R2plus1D_18_MotionNet
from src.model.dropout_v4_0_10_R2plus1D_18_MotionNet import dropout_v4_0_10_R2plus1D_18_MotionNet
from src.model.dropout_v4_0_25_R2plus1D_18_MotionNet import dropout_v4_0_25_R2plus1D_18_MotionNet
from src.model.dropout_v4_0_50_R2plus1D_18_MotionNet import dropout_v4_0_50_R2plus1D_18_MotionNet
from src.model.dropout_v4_0_75_R2plus1D_18_MotionNet import dropout_v4_0_75_R2plus1D_18_MotionNet

# for finding lv seg borders
import cv2 as cv

# for storing vector snapshots
import copy

# from src.visualization_utils import categorical_dice

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import random
import pickle
import time

tic, toc = (time.time, time.time)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############

epoch = 1
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


with open("fold_indexes/stanford_valid_sampled_indices", "rb") as infile:
    valid_mask = pickle.load(infile)
infile.close()

# test_dataset = EchoNetDynamicDataset(split='test', clip_length="full", raise_for_es_ed=False, period=1)

# test_loader = DataLoader(test_dataset, batch_size=batch_size, 
#                               num_workers=num_workers,
#                               shuffle=False, pin_memory=("cuda"),
#                               worker_init_fn=worker_init_fn_valid )

valid_dataset = EchoNetDynamicDataset(split='val', subset_indices=valid_mask, period=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, 
                              num_workers=num_workers,
                              shuffle=False, pin_memory=("cuda"),
                              worker_init_fn=worker_init_fn_valid
                             )


#################
input_data_loader = valid_dataloader
#################
model_save_path = f"save_models/{model_name}"
    
if model_name == 'Original_Pretrained_R2plus1DMotionSegNet.pth':
    model_template_obj = R2plus1D_18_MotionNet()
elif model_name == 'dropout_v2_0_00_R2plus1DMotionSegNet.pth':
    model_template_obj = dropout_v2_0_00_R2plus1D_18_MotionNet()
elif model_name == 'dropout_v2_0_10_R2plus1DMotionSegNet.pth':
    model_template_obj = dropout_v2_0_10_R2plus1D_18_MotionNet()
elif model_name == 'dropout_v2_0_25_R2plus1DMotionSegNet.pth':
    model_template_obj = dropout_v2_0_25_R2plus1D_18_MotionNet()


elif model_name == "dropout_v3_0_00_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v3_0_00_R2plus1D_18_MotionNet()
elif model_name == "dropout_v3_0_10_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v3_0_10_R2plus1D_18_MotionNet()
elif model_name == "dropout_v3_0_25_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v3_0_25_R2plus1D_18_MotionNet()
elif model_name == "dropout_v4_0_00_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v4_0_00_R2plus1D_18_MotionNet()
elif model_name == "dropout_v4_0_10_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v4_0_10_R2plus1D_18_MotionNet()
elif model_name == "dropout_v4_0_25_R2plus1DMotionSegNet.pth":
    model_template_obj = dropout_v4_0_25_R2plus1D_18_MotionNet()


model = torch.nn.DataParallel(model_template_obj)

model.to("cuda")
torch.cuda.empty_cache()
model.load_state_dict(torch.load(model_save_path)["model"])
print(f'{model_name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')
model.eval();
###########################
from src.strain import *

frames_to_study = {}

class DiceLoss(nn.Module):
    """
        Dice loss
        See here: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=68471013&cellId=4
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def generate_2dmotion_field(x, offset):
    # Qin's code for joint_motion_seg learning works fine on our purpose too
    # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
    
    global frames_to_study
    
    frames_to_study['generate_2dmotion_field'] = {}
    # frames_to_study['generate_2dmotion_field']
    
    # shape: (B, C, H, W)
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    # split (4, 2, 112, 112) into 2 (4, 112, 112)
    # presumably to keep the LV and NOT LV motions separate?
    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h
    
    offsets = torch.stack((offset_h, offset_w), 3)

    return offsets


def motion_seg_loss(label_ed, label_es, ed_index, es_index, motion_output, seg_softmax, 
                    start=0, end=32, seg_criterion=DiceLoss()):
    """
        SGS loss that spatially transform the true ED and true ES fully forward to the end of video
        and backward to the beginning. Then, compare the forward and backward transformed pseudo labels with
        segmentation at all frames.
    """
    flow_source = convert_to_1hot(label_ed, 2)
    loss_forward = 0
    OTS_loss = 0
    OTS_criterion = DiceLoss()
    
    global frames_to_study
    
    frames_to_study['one hot label_ed'] = flow_source
    
    # Forward from ed to the end of video
    print('Forward from ed to the end of video')
    frames_to_study['Forward from ed to the end of video'] = {}
    frames_to_study['Forward from ed to the end of video']['flow_sources'] = []
    frames_to_study['Forward from ed to the end of video']['motion_fields'] = []
    frames_to_study['Forward from ed to the end of video']['forward_motions'] = []

    
    for frame_index in range(ed_index, end - 1):
        forward_motion = motion_output[:, :2, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, forward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        if frame_index == (es_index - 1):
            one_hot_ES = convert_to_1hot(label_es, 2)
            OTS_loss += OTS_criterion(next_label, one_hot_ES)
        else:
            loss_forward += seg_criterion(seg_softmax[:, :, frame_index + 1, ...], next_label)
        flow_source = next_label
        
        frames_to_study['Forward from ed to the end of video']['forward_motions'].append([frame_index, forward_motion])
        frames_to_study['Forward from ed to the end of video']['motion_fields'].append([frame_index, motion_field])
        frames_to_study['Forward from ed to the end of video']['flow_sources'].append([frame_index, flow_source])
        
    # Forward from es to the end of video
    frames_to_study['Forward from es to the end of video'] = []
    print('Forward from es to the end of video')
    flow_source = convert_to_1hot(label_es, 2)
    for frame_index in range(es_index, end - 1):
        forward_motion = motion_output[:, :2, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, forward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')

        loss_forward += seg_criterion(seg_softmax[:, :, frame_index + 1, ...], next_label)
        flow_source = next_label
        
        
        frames_to_study['Forward from es to the end of video'].append([frame_index, flow_source])

        
    flow_source = convert_to_1hot(label_es, 2)
    loss_backward = 0
    
    frames_to_study['one hot label_es'] = flow_source
    
    # Backward from es to the beginning of video
    frames_to_study['Backward from es to the beginning of video'] = []
    print('Backward from es to the beginning of video')

    for frame_index in range(es_index, start, -1):
        backward_motion = motion_output[:, 2:, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, backward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        if frame_index == ed_index + 1:
            one_hot_ED = convert_to_1hot(label_ed, 2)
            OTS_loss += OTS_criterion(next_label, one_hot_ED)
        else:
            loss_backward += seg_criterion(seg_softmax[:, :, frame_index - 1, ...], next_label)
        flow_source = next_label
        
        frames_to_study['Backward from es to the beginning of video'].append([frame_index, flow_source])
    
    
    flow_source = convert_to_1hot(label_ed, 2)
    
    
    # Backward from ed to the beginning of video
    frames_to_study['Backward from ed to the beginning of video'] = []
    print('Backward from ed to the beginning of video')

    
    for frame_index in range(ed_index, start, -1):
        backward_motion = motion_output[:, 2:, frame_index,...]
        motion_field = generate_2dmotion_field(flow_source, backward_motion)
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        
        loss_backward += seg_criterion(seg_softmax[:, :, frame_index - 1, ...], next_label)
        flow_source = next_label
        
        frames_to_study['Backward from ed to the beginning of video'].append([frame_index, flow_source])

        
    # Averaging the resulting dice
    flow_loss = (loss_forward + loss_backward) / ((motion_output.shape[2] - 2) * 2)
    OTS_loss = OTS_loss / 2 
    
    return flow_loss, OTS_loss



def deformation_motion_loss(source_videos, motion_field):
    """
        OTA loss for motion tracking on echocardiographic frames
    """
    mse_criterion = nn.MSELoss()
    mse_loss = 0
    smooth_loss = 0
    
    # Deform both forward and backward from beginning to the end of video clip 
    for index in range(source_videos.shape[2] - 1):
        forward_motion = motion_field[:, :2, index,...]
        backward_motion = motion_field[:, 2:, index + 1,...]
        
        grid_forward = generate_2dmotion_field(source_videos[:, :, index,...], forward_motion)
        grid_backward = generate_2dmotion_field(source_videos[:, :, index + 1,...], backward_motion)
        
        pred_image_forward = F.grid_sample(source_videos[:, :, index,...], grid_forward, 
                                           align_corners=False, padding_mode='border')
        pred_image_backward = F.grid_sample(source_videos[:, :, index + 1,...], grid_backward, 
                                            align_corners=False, padding_mode='border')
        
        mse_loss += mse_criterion(source_videos[:, :, index + 1,...], pred_image_forward)
        mse_loss += mse_criterion(source_videos[:, :, index,...], pred_image_backward)
        
        smooth_loss += huber_loss(forward_motion)
        smooth_loss += huber_loss(backward_motion)
    
    # Averaging the resulting loss
    return (0.005 * smooth_loss + mse_loss) / 2 / (source_videos.shape[2] - 1)


# altered from the original test function, removed the param optimizer bc I don't see it being used
def test(epoch, test_loader, model):
    
    global frames_to_study
    
    model.eval()
    epoch_loss = []
    ed_lv_dice = 0
    es_lv_dice = 0
    
    for batch_idx, batch in enumerate(test_loader, 1):
        filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label = batch[1]
        with torch.no_grad():
            video_clips = torch.Tensor(batch[0])
            video_clips = video_clips.type(Tensor)

        # Get the motion tracking output from the motion tracking head using the feature map
        segmentation_output, motion_output = model(video_clips)
        
        frames_to_study['raw_seg_out'] = segmentation_output
        frames_to_study['raw_motion_out'] = motion_output
        
        frames_to_study['ed_index'] = ed_index
        frames_to_study['es_index'] = es_index
        
        frames_to_study['delta_ed_es'] = es_index - ed_index
        frames_to_study['filename'] = filename
        
        loss = 0
        deform_loss = deformation_motion_loss(video_clips, motion_output)
        loss += deform_loss

        segmentation_loss = 0
        motion_loss = 0
        print(f'for i in range(video_clips.shape[0]={video_clips.shape[0]}')
        for i in range(video_clips.shape[0]):
            print(f'i={i}')
            label_ed = np.expand_dims(ed_label.numpy(), 1).astype("int")
            label_es = np.expand_dims(es_label.numpy(), 1).astype("int")

            label_ed = label_ed[i]
            label_es = label_es[i]

            label_ed = np.expand_dims(label_ed, 0)
            label_es = np.expand_dims(label_es, 0)

            motion_one_output = motion_output[i].unsqueeze(0)
            segmentation_one_output = segmentation_output[i].unsqueeze(0)

            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]

            print('Enter: motion_seg_loss')
            segmentation_one_loss, motion_one_loss = motion_seg_loss(label_ed, label_es, 
                                                                     ed_one_index, es_one_index, 
                                                                     motion_one_output, segmentation_one_output, 
                                                                     0, video_clips.shape[2], 
                                                                     F.binary_cross_entropy_with_logits)
            print('Leave: motion_seg_loss')
            segmentation_loss += segmentation_one_loss
            motion_loss += motion_one_loss
            
            frames_to_study['segmentation_loss'] = segmentation_loss
            frames_to_study['motion_loss'] = motion_loss
            
            break # just do one iteration
            
        return # ignore other stuff, just focus on the loss that uses the motion tracking.
            
        loss += (segmentation_loss / video_clips.shape[0])
        loss += (motion_loss / video_clips.shape[0])
        
        ed_segmentations = torch.Tensor([]).type(Tensor)
        es_segmentations = torch.Tensor([]).type(Tensor)
        for i in range(len(ed_clip_index)):
            ed_one_index = ed_clip_index[i]
            es_one_index = es_clip_index[i]
            
            ed_seg = segmentation_output[i, :, ed_one_index].unsqueeze(0)
            ed_segmentations = torch.cat([ed_segmentations, ed_seg])
            
            es_seg = segmentation_output[i, :, es_one_index].unsqueeze(0)
            es_segmentations = torch.cat([es_segmentations, es_seg])
            
            
        ed_es_seg_loss = 0
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(ed_segmentations, 
                                                             convert_to_1hot(np.expand_dims(ed_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        
        ed_es_seg_loss += F.binary_cross_entropy_with_logits(es_segmentations, 
                                                             convert_to_1hot(np.expand_dims(es_label.numpy().astype("int"), 1), 2), 
                                                             reduction="mean") 
        ed_es_seg_loss /= 2
        
        loss += ed_es_seg_loss
        
        epoch_loss.append(loss.item())
        
        ed_segmentation_argmax = torch.argmax(ed_segmentations, 1).cpu().detach().numpy()
        es_segmentation_argmax = torch.argmax(es_segmentations, 1).cpu().detach().numpy()
        
        ed_lv_dice += categorical_dice(ed_segmentation_argmax, ed_label.numpy(), 1)
        es_lv_dice += categorical_dice(es_segmentation_argmax, es_label.numpy(), 1)
    
    print("-" * 30 + "Validation" + "-" * 30)
    print("\nED LV: {:.3f}".format(ed_lv_dice / batch_idx))
    print("ES LV: {:.3f}".format(es_lv_dice / batch_idx))
        
        # Printing the intermediate training statistics
        
    print('\nValid set: Average loss: {:.4f}\n'.format(np.mean(epoch_loss)))
    
    return epoch_loss


Tensor = torch.cuda.FloatTensor

epoch_loss = test(epoch, input_data_loader, model)