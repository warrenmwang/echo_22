import argparse
ap = argparse.ArgumentParser(description="test")
ap.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
args = ap.parse_args()

model_names = args.list # we are going to assume that only one model is used each time for this script

# output csv file name and location
csv_file = f"./dev/quantifying-performance/csvs/{model_names[0]}.csv"

# leave false to go thru all test videos, set to true and adjust custom number of videos for testing purposes
USE_CUSTOM_NUM_VIDS_TO_GO_THRU = False
CUSTOM_NUM_VIDS = 1

import os
os.chdir("/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking")

import sys
sys.path.append('/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking')


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

# v1 dropout, not in place dropout
from src.model.dropout_0_10_R2plus1D_18_MotionNet import dropout_0_10_R2plus1D_18_MotionNet
from src.model.dropout_0_25_R2plus1D_18_MotionNet import dropout_0_25_R2plus1D_18_MotionNet
from src.model.dropout_0_50_R2plus1D_18_MotionNet import dropout_0_50_R2plus1D_18_MotionNet
from src.model.dropout_0_75_R2plus1D_18_MotionNet import dropout_0_75_R2plus1D_18_MotionNet

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

# import all of our strain functions
from src.strain import *

tic, toc = (time.time, time.time)

batch_size = 4
num_workers = max(4, cpu_count()//2)


def worker_init_fn_valid(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def worker_init_fn(worker_id):
    # See here: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    # and the original post of the problem: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817373837
    worker_seed = torch.initial_seed() % 2 ** 32
    print(f'worker_seed: {worker_seed}')
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

# full_dataset = EchoNetDynamicDataset(split='val', clip_length="full", subset_indices=valid_mask, period=1)
test_dataset = EchoNetDynamicDataset(split='test', clip_length="full", raise_for_es_ed=False, period=1)
# random_test_dataset = EchoNetDynamicDataset(split='test', clip_length=32, raise_for_es_ed=True, period=1)

def get_all_possible_start_points(ed_index, es_index, video_length, clip_length):
    assert es_index - ed_index > 0, "not a ED to ES clip pair"
    possible_shift = clip_length - (es_index - ed_index)
    allowed_right = video_length - es_index
    if allowed_right < possible_shift:
        return np.arange(ed_index - possible_shift + 1, video_length - clip_length + 1)
    if possible_shift < 0:
        return np.array([ed_index])
    elif ed_index < possible_shift:
        return np.arange(ed_index + 1)
    else:
        return np.arange(ed_index - possible_shift + 1, ed_index + 1)
    

# from queue import SimpleQueue as squeue
def EDESpairs(diastole, systole):
    dframes = np.sort(np.array(diastole))
    sframes = np.sort(np.array(systole))
    clips = []
    
    inds = np.searchsorted(dframes, sframes, side='left')
    for i, sf in enumerate(sframes):
        if inds[i] == 0: # no prior diastolic frames for this sf
            continue
        best_df = diastole[inds[i]-1] # diastole frame nearest this sf.
        if len(clips) == 0 or best_df != clips[-1][0]:
            clips.append((best_df, sf))
            
    return clips

loaded_in_models = []
device = "cuda" if torch.cuda.is_available() else 'cpu'

# load in the trained model from the .pth file containin the model's weights and biases
for model_name in model_names:
    model_save_path = f"save_models/{model_name}"     
    
    if model_name == 'Original_Pretrained_R2plus1DMotionSegNet.pth':
        model_template_obj = R2plus1D_18_MotionNet()
    elif model_name == 'retrain_original_R2plus1DMotionSegNet.pth':
        model_template_obj = R2plus1D_18_MotionNet()
    elif model_name == 'dropout_v2_0_00_R2plus1DMotionSegNet.pth':
        model_template_obj = dropout_v2_0_00_R2plus1D_18_MotionNet()
    elif model_name == 'dropout_v2_0_10_R2plus1DMotionSegNet.pth':
        model_template_obj = dropout_v2_0_10_R2plus1D_18_MotionNet()
    elif model_name == 'dropout_v2_0_25_R2plus1DMotionSegNet.pth':
        model_template_obj = dropout_v2_0_25_R2plus1D_18_MotionNet()
    elif model_name == 'dropout_v2_0_50_R2plus1DMotionSegNet.pth':
        model_template_obj = dropout_v2_0_50_R2plus1D_18_MotionNet()
    elif model_name == 'dropout_v2_0_75_R2plus1DMotionSegNet.pth':
        model_template_obj = dropout_v2_0_75_R2plus1D_18_MotionNet()
        
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
    model.to(device)
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_save_path)["model"])
    print(f'{model_name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')
    model.eval();
    
    loaded_in_models.append((model_name, model))

print(len(loaded_in_models))

########################## Start of defining functions ####################################

def divide_to_consecutive_clips(video, clip_length=32, interpolate_last=False):
    '''
    i think this divides your say 200 frame video into the minimum number 
    of clips of length 32 that you want:
    example: 200/32 = 6.25, so you would need minimum of 7 32 frame clips
    to cover the entire range of this 200 frame video
    question: what do you fill the pixel values in the unnecessary/extra frames in the last 
    32 frame clip? 
    '''
    
    # print('enter divide_to_consecutive_clips')

    source_video = video.copy()
    video_length = video.shape[1]
    left = video_length % clip_length
    if left != 0 and interpolate_last:
        source_video = torch.Tensor(source_video).unsqueeze(0)
        source_video = F.interpolate(source_video, size=(int(np.round(video_length / clip_length) * clip_length), 112, 112),
                                     mode="trilinear", align_corners=False)
        source_video = source_video.squeeze(0).squeeze(0)
        source_video = source_video.numpy()

    videos = np.empty(shape=(1, 3, clip_length, 112, 112))

    # for debugging
    all_starts = []

    for start in range(0, int(clip_length * np.round(video_length / clip_length)), clip_length):
        all_starts.append(start)

        one_clip = source_video[:, start: start + clip_length]
        one_clip = np.expand_dims(one_clip, 0)
        videos = np.concatenate([videos, one_clip])

    # print(f'all_starts: {all_starts}')
    # print('leave divide_to_consecutive_clips')

    return videos[1:]

def segment_a_video_with_fusion_custom(curr_model, log_file, video, interpolate_last=True, step=1, num_clips=10, 
                                fuse_method="simple", class_list=[0, 1]):
    '''
    this will spit out segmentations and motion trackings
    information about a video from all of its frames consecutively 
    '''
    
    # print('enter segment_a_video_with_fusion_custom')
    
    # calculate how many clips we will actually make 
    if video.shape[1] < 32 + num_clips * step:
        num_clips = (video.shape[1] - 32) // step
    if num_clips < 0:
        log_file.write("Video is too short\n")
        num_clips = 1
        
    # print(f'num_clips: {num_clips}')
        
    all_consecutive_clips = []

    for shift_dis in range(0, num_clips * step, step):
        shifted_video = video[:, shift_dis:]
        consecutive_clips = divide_to_consecutive_clips(shifted_video, interpolate_last=interpolate_last)
        all_consecutive_clips.append(consecutive_clips)

    all_consecutive_clips = np.array(all_consecutive_clips)
    
    # print(f'all_consecutive_clips.shape: {all_consecutive_clips.shape}')
    # print(f'len(all_consecutive_clips): {len(all_consecutive_clips)}')
    
    all_segmentations = []
    all_motions = []

    for i in range(len(all_consecutive_clips)):
        consecutive_clips = all_consecutive_clips[i]
        segmentation_outputs = np.empty(shape=(1, 2, 32, 112, 112))
        
        motion_outputs = np.empty(shape=(1,4,32,112,112))

        for i in range(consecutive_clips.shape[0]):
            one_clip = np.expand_dims(consecutive_clips[i], 0)
            
            segmentation_output, motion_output = curr_model(torch.Tensor(one_clip))
            
            # print(f'segmentation_output BEFORE softmax: {segmentation_output.shape}')
            
            # seg -> softmax func
            segmentation_output = F.softmax(segmentation_output, 1)
            
            # print(f'segmentation_output AFTER softmax: {segmentation_output.shape}')
            
            # save
            segmentation_outputs = np.concatenate([segmentation_outputs, segmentation_output.cpu().detach().numpy()])
            
            # don't want to apply softmax to motion out, just save
            motion_outputs = np.concatenate([motion_outputs, motion_output.cpu().detach().numpy()])     
        
        # don't need extra pre dim 1
        segmentation_outputs = segmentation_outputs[1:]
        motion_outputs = motion_outputs[1:]
        
        all_motions.append(motion_outputs)
        all_segmentations.append(segmentation_outputs)
    
#     print(f'len(all_segmentations): {len(all_segmentations)}')
#     print(f'len(all_motions): {len(all_motions)}')
    
#     print(f'before all_segmentations[0].shape transpose and reshape: {all_segmentations[0].shape}')
#     print(f'before mot transpose and reshape: {all_motions[0].shape}')

    # what does this do? this looks like it transposes and reshapes the things in the segmentations
    for i in range(len(all_segmentations)):
        all_segmentations[i] = all_segmentations[i].transpose([1, 0, 2, 3, 4])
        all_segmentations[i] = all_segmentations[i].reshape(2, -1, 112, 112)
    
    # TODO? do I need to do the same thing in the for loop above for the motion trackings?

    
    # print(f'after trans/reshape all_segmentations[0].shape: {all_segmentations[0].shape}')
    # print(f'no change to: all_motions[0].shape: {all_motions[0].shape}')
    
    # what are the shapes of the things in the segmentations
    # print(f'before interpolate all_segmentations[0].shape: {all_segmentations[0].shape}')


    ############### Interpolate results back to original length ###############
    # segmentation
    all_interpolated_segmentations = []
    for i in range(0, len(all_consecutive_clips)):
        video_clip = video[:, i * step:]
        if interpolate_last and (video_clip.shape[1] % 32 != 0):
            interpolated_segmentations = torch.Tensor(all_segmentations[i]).unsqueeze(0)
            interpolated_segmentations = F.interpolate(interpolated_segmentations, size=(video_clip.shape[1], 112, 112), 
                                                       mode="trilinear", align_corners=False)
            interpolated_segmentations = interpolated_segmentations.squeeze(0).numpy()
            all_interpolated_segmentations.append(np.argmax(interpolated_segmentations, 0))
        else:
            all_interpolated_segmentations.append(np.argmax(all_segmentations[i], 0))

    fused_segmentations = [all_interpolated_segmentations[0][0]]
    
    # print('after interpolate')
    # print(f'len(fused_segmentations: {len(fused_segmentations)}')
    # print(f'fused_segmentations[0].shape: {fused_segmentations[0].shape}')
    
    # TODO: motion tracking, 
    # problem: what shape does the motion tracking object need to be in for the following code to work?
    # problem: i don't know what interpolation is and how Yida is doing this. What is Label Fusion?
    # solution: 
    # print('try interpolate motion tracking')
    
#     all_interpolated_motions = []
#     for i in range(0, len(all_consecutive_clips)):
#         video_clip = video[:, i * step:]
#         if interpolate_last and (video_clip.shape[1] % 32 != 0):
#             interpolated_motions = torch.Tensor(all_motions[i]).unsqueeze(0)
#             interpolated_motions = F.interpolate(interpolated_motions, size=(video_clip.shape[1], 112, 112), 
#                                                        mode="trilinear", align_corners=False)
#             interpolated_motions = interpolated_motions.squeeze(0).numpy()
#             all_interpolated_motions.append(np.argmax(interpolated_motions, 0))
#         else:
#             all_interpolated_motions.append(np.argmax(all_motions[i], 0))

#     fused_motions = [all_interpolated_motions[0][0]]
    
    # print(f'after interpolate fused_motions[0].shape: {fused_motions[0].shape}')
    
    ############### ############### ############### ###############
    
    ############### Align Frames ###############
    for i in range(1, video.shape[1]):
        if step - 1 < i:
            # seg
            images_to_fuse_segmentations = []
            for index in range(min(i, len(all_interpolated_segmentations))):
                if i - index * step < 0:
                    break
                images_to_fuse_segmentations.append(itk.GetImageFromArray(all_interpolated_segmentations[index][i - index * step].astype("uint8"),
                                                            isVector=False))
            
            # mot
            # images_to_fuse_motions = []
            # for index in range(min(i, len(all_interpolated_motions))):
            #     if i - index * step < 0:
            #         break
            #     images_to_fuse_motions.append(itk.GetImageFromArray(all_interpolated_motions[index][i - index * step].astype("uint8"),
                                                            # isVector=False))

    ############### ############### ###############
    ############### Do the fusion ###############
            # seg
            if len(images_to_fuse_segmentations) <= 1:
                fused_segmentations.append(itk.GetArrayFromImage(images_to_fuse_segmentations[0]))
            else:
                fused_image = fuse_images(images_to_fuse_segmentations, fuse_method, class_list=class_list)
                # If using SIMPLE, the fused image might be in type "float"
                # So convert it to uint
                fused_segmentations.append(itk.GetArrayFromImage(fused_image).astype("uint8"))
                
            # mot
            # if len(images_to_fuse_motions) <= 1:
            #     fused_motions.append(itk.GetArrayFromImage(images_to_fuse_motions[0]))
            # else:
            #     fused_image = fuse_images(images_to_fuse_motions, fuse_method, class_list=class_list)
            #     # If using SIMPLE, the fused image might be in type "float"
            #     # So convert it to uint
            #     fused_motions.append(itk.GetArrayFromImage(fused_image).astype("uint8"))

    ############### ############### ###############
                
    fused_segmentations = np.array(fused_segmentations)
    
    # fused_motions = np.array(fused_motions)
    
    # print(f'fused_segmentations FINAL: {fused_segmentations.shape}')
    
    # print(f'fused_motions.shape: {fused_motions.shape}')
    
    return fused_segmentations


def compute_ef_using_putative_clips(fused_segmentations, test_pat_index, log_file):
    '''
    computes ef from the segmentation outputs from the consecutively cut 32 frame clips ? or is that done by the other compute ef
    function :/... what's putative clips vs reported clips?
    '''
    
    size = np.sum(fused_segmentations, axis=(1, 2)).ravel()
    _05cut, _85cut, _95cut = np.percentile(size, [5, 85, 95]) 

    trim_min = _05cut
    trim_max = _95cut
    trim_range = trim_max - trim_min
    systole = find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0]
    diastole = find_peaks(size, distance=20, prominence=(0.50 * trim_range))[0]

    # keep only real diastoles..
    diastole = [x for x in diastole if size[x] >= _85cut]
    # Add first frame
    if np.mean(size[:3]) >= _85cut:
        diastole = [0] + diastole
    diastole = np.array(diastole)

    clip_pairs = EDESpairs(diastole, systole)

    one_array_of_segmentations = fused_segmentations.reshape(-1, 112, 112)

    predicted_efs = []

    for i in range(len(clip_pairs)):
        output_ED = one_array_of_segmentations[clip_pairs[i][0]]
        output_ES = one_array_of_segmentations[clip_pairs[i][1]]
        
        length_ed, radius_ed = get2dPucks((output_ED == 1).astype('int'), (1.0, 1.0))
        length_es, radius_es = get2dPucks((output_ES == 1).astype('int'), (1.0, 1.0))

        edv = np.sum(((np.pi * radius_ed * radius_ed) * length_ed / len(radius_ed)))
        esv = np.sum(((np.pi * radius_es * radius_es) * length_es / len(radius_es)))

        ef_predicted = (edv - esv) / edv * 100
        
        if ef_predicted < 0:
            log_file.write("Negative EF at patient:{:04d}".format(test_pat_index))
            continue

        predicted_efs.append(ef_predicted)

    return predicted_efs


def compute_ef_using_reported_clip(segmentations, ed_index, es_index):
    output_ED = segmentations[ed_index]
    output_ES = segmentations[es_index]

    lv_ed_dice = categorical_dice((output_ED), ed_label, 1)
    lv_es_dice = categorical_dice((output_ES), es_label, 1)

    length_ed, radius_ed = get2dPucks((output_ED == 1).astype('int'), (1.0, 1.0))
    length_es, radius_es = get2dPucks((output_ES == 1).astype('int'), (1.0, 1.0))
    
    edv = np.sum(((np.pi * radius_ed * radius_ed) * length_ed / len(radius_ed)))
    esv = np.sum(((np.pi * radius_es * radius_es) * length_es / len(radius_es)))

    ef_predicted = (edv - esv) / edv * 100
    
    return ef_predicted, lv_ed_dice, lv_es_dice

########################## End of defining functions ####################################


with open(csv_file, "a") as file:
    # csv header
    file.write("Video Index,Predicted EF,True EF,Seg GLS,Warp GLS,True GLS,Seg ED Dice,Seg ES Dice,Warp ED Dice,Warp ES Dice\n")
    
    for j in range(len(loaded_in_models)):
        model = loaded_in_models[j][1]
        model_name = loaded_in_models[j][0]

        ###########
        patient_filename = []

        num_clips = 5
        step = 1
        interpolate_last = True
        fuse_method = "simple"
        class_list = [0, 1]

        start = time.time()
        
        # choose how many videos you want to calculate EF for 
        if USE_CUSTOM_NUM_VIDS_TO_GO_THRU:
            NUM_VIDS_TO_GO_THRU = CUSTOM_NUM_VIDS
        else:
            NUM_VIDS_TO_GO_THRU = len(test_dataset)

        # loop for number of videos from the test dataset that we want to go over
        for i in range(NUM_VIDS_TO_GO_THRU):
        # for i in range(, len(test_dataset)):
            test_pat_index = i
            try:
                video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
            except:
                file.write("Get exception when trying to read the video from patient:{:04d}\n".format(i))
                continue
            
            try:
                # manual adjustment for a known video index (i didn't write this)
                # it looks like it skips the first 80 frames ?
                if test_pat_index == 1053:
                    video = video[:, :80]

                segmentations = segment_a_video_with_fusion_custom(model, file, video, interpolate_last=interpolate_last, 
                                                            step=step, num_clips=num_clips,
                                                            fuse_method=fuse_method, class_list=class_list)

                # print(f'segmentations.shape: {segmentations.shape}') # (248, 112, 112)
                # print(f'motions.shape: {motions.shape}') # (5, 8, 4, 32, 112, 112)

                # motions = motions[0] # (8, 4, 32, 112, 112)
                # tmp = motions.shape
                # np.resize(motions, (tmp[0] * tmp[2], tmp[1], tmp[3], tmp[4]))


                # REMOVE ME later
                # sys.exit(0)


                # these should all be shape (112, 112) and have its pixel values either 1 or 0, 1 for lv, 0 for not lv
                ed_from_seg = segmentations[ed_index] 
                es_from_seg = segmentations[es_index] 

                # get ed and es frames from warping
                es_from_warp, ed_from_warp = get_warped_ed_es_frames(test_pat_index = test_pat_index, 
                                                                     test_dataset = test_dataset, 
                                                                     model = model)


                # ed_from_warp = None # TODO
                # es_from_warp = None # TODO

                # EFs
                predicted_efs = compute_ef_using_putative_clips(segmentations, test_pat_index=test_pat_index, log_file=file)

                # Dice
                # segmentation & GT
                _, ed_dice, es_dice = compute_ef_using_reported_clip(segmentations, ed_index, es_index)
                # warps & GT
                warp_ed_dice = categorical_dice(ed_from_warp, ed_label , k=1, epsilon=1e-5)
                warp_es_dice = categorical_dice(es_from_warp, es_label , k=1, epsilon=1e-5)

                # GLS
                # ground truth
                ground_truth_strain = images_to_strain(ed_label, es_label)

                # segmentation
                seg_out_strain = images_to_strain(ed_from_seg, es_from_seg)

                # just warp (with base from ED/ES from seg out)
                warp_strain = images_to_strain(ed_from_warp, es_from_warp)


                # skip this video if we get weird results (should investigate deeper into this to know why.)
                if len(predicted_efs) == 0:
                    file.write("Cannot identify clips at patient:{:04d}\n".format(test_pat_index))
                    continue

                if np.isnan(np.nanmean(predicted_efs)):
                    file.write("Cannot identify clips at patient:{:04d}\n".format(test_pat_index))
                    continue

                # save as csv style
                file.write(f'{test_pat_index},{np.nanmean(predicted_efs)},{EF},{seg_out_strain},{warp_strain},{ground_truth_strain},{ed_dice},{es_dice},{warp_ed_dice},{warp_es_dice}\n')
            
            except Exception as e:
                print(f'Got "{e}" at index: {test_pat_index}')

        end = time.time()

print(f"Used time = {(end - start) // 60:.0f} mins {(end - start) % 60:.0f} secs\n")