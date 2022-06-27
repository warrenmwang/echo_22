# model_names = ['Original_Pretrained_R2plus1DMotionSegNet_model.pth', 
#                'retrain_original_R2plus1DMotionSegNet_model.pth', 
#                'dropout_v2_0_00_R2plus1DMotionSegNet_model.pth',
#                'dropout_v2_0_10_R2plus1DMotionSegNet_model.pth',
#                'dropout_v2_0_25_R2plus1DMotionSegNet_model.pth',
#                'dropout_v2_0_50_R2plus1DMotionSegNet_model.pth',
#                'dropout_0_10_R2plus1DMotionSegNet_model.pth',
#                'dropout_0_25_R2plus1DMotionSegNet_model.pth',
#                'dropout_0_50_R2plus1DMotionSegNet_model.pth',
#               ]
import argparse
ap = argparse.ArgumentParser(description="test")
ap.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
args = ap.parse_args()

model_names = args.list


logfile = open("./warren-random/longitudinal_strain/NEW-strains-log.txt", "a")

import os
os.chdir("/home/wang/workspace/JupyterNoteBooksAll/fully-automated-multi-heartbeat-echocardiography-video-segmentation-and-motion-tracking")
# print(os.getcwd())

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

# for finding lv seg borders
import cv2 as cv


# from src.visualization_utils import categorical_dice
# %matplotlib widget
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

start_time = tic()

def strain_value(l_0, l_i):
    '''
    inputs: l_0, l_i -- original length and new length at some time point, respectively
    output: e -- strain value (positive for elongation, negative for compressing/shortening) as a percentage (e.g. output 0.155 == 15.5 %)
    
    examples: 
        l_i = 10
        l_0 = 5
        e == (10 - 5) / 5 = 1, factor of lengthening relative to original value
        
        l_i = 5
        l_0 = 5
        e == (5 - 5) / 5 = 0, no strain
    '''
    return (l_i - l_0) / l_0


def rmse(x, y):
    ''' return root mean square error difference between the two values passed in'''
    return np.sqrt((x - y) ** 2)

def fromSegOutToEDESLVMasksAndBorders(all_segmentation_outputs, es_index, ed_index, model_num = 0):
    '''
    input: all_segmentation_outputs -- the segmentation outputs from the model wrapped in a list of len number of models that we have
    output: [[ed lv seg, es lv seg], [ed lv seg border, es lv seg border]] all will be 112 x 112 images which pixels of value 0 or 255
    '''
    
    # how many frames ahead is the ES frame relative to the ED frame?
    delta_ed_es_frames = es_index - ed_index

    
    clip_to_grab_index = -1   # grab the last clip where ED frame is the first frame
    # model_num = model_num     # which model's output to grab (0 if just using 1 model)
    clip_with_ed_frame_index_0 = all_segmentation_outputs[model_num][clip_to_grab_index] 

    foo = np.argmax(clip_with_ed_frame_index_0, axis=0)
    foo_times_255 = 255 * foo

    ed_lv_seg = foo_times_255[0]
    es_lv_seg = foo_times_255[delta_ed_es_frames]
    
    
    image_8bit = np.uint8(foo_times_255)
    ed_im = image_8bit[0]
    es_im = image_8bit[delta_ed_es_frames]

    
    ret, thresh = cv.threshold(ed_im, 127, 255, 0)
    ed_contours, ed_hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ret, thresh = cv.threshold(es_im, 127, 255, 0)
    es_contours, es_hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    ed_blank = np.zeros((112,112))
    cv.drawContours(ed_blank, ed_contours, -1, (255,255,255), 1)

    es_blank = np.zeros((112,112))
    cv.drawContours(es_blank, es_contours, -1, (255,255,255), 1)
    
    return [[ed_lv_seg, es_lv_seg], [ed_blank, es_blank]]
    
def groundTruthBoundaries(ed_label, es_label):
    '''
    input: 
        ed_label, es_label -- 112 x 112 manual segmented ed/es lv answers (input values are either 0 or 1)
        shapes: (112, 112)
    output: [[ed label, es label], [ed label border, es label border]]
    '''
    ed_im = np.uint8(ed_label * 255)
    es_im = np.uint8(es_label * 255)

    ret, thresh = cv.threshold(ed_im, 127, 255, 0)
    ed_contours, ed_hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ret, thresh = cv.threshold(es_im, 127, 255, 0)
    es_contours, es_hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    ed_blank = np.zeros((112,112))
    cv.drawContours(ed_blank, ed_contours, -1, (255,255,255), 1)

    es_blank = np.zeros((112,112))
    cv.drawContours(es_blank, es_contours, -1, (255,255,255), 1)

    return [[ed_im, es_im], [ed_blank, es_blank]]


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

full_dataset = EchoNetDynamicDataset(split='val', clip_length="full", raise_for_es_ed=False, subset_indices=valid_mask, period=1)
test_dataset = EchoNetDynamicDataset(split='test', clip_length="full", raise_for_es_ed=False, period=1)

logfile.write(f'test dataset length: {len(test_dataset)}\n')


def divide_to_consecutive_clips(video, clip_length=32, interpolate_last=False):
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

    for start in range(0, int(clip_length * np.round(video_length / clip_length)), clip_length):
        one_clip = source_video[:, start: start + clip_length]
        one_clip = np.expand_dims(one_clip, 0)
        videos = np.concatenate([videos, one_clip])
    return videos[1:]


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
    
    
loaded_in_models = []

for model_name in model_names:
    model_save_path = f"save_models/{model_name}"
    
#     # original model
#     if model_name == "Original_Pretrained_R2plus1DMotionSegNet_model.pth":
#          model = torch.nn.DataParallel(R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == "retrain_original_R2plus1DMotionSegNet_model.pth":
#         model = torch.nn.DataParallel(R2plus1D_18_MotionNet(), device_ids = [1, 0])
        
#     # altered models
#     if model_name == "dropout_0_75_R2plus1DMotionSegNet_model.pth":
#         model = torch.nn.DataParallel(dropout_0_75_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == "dropout_0_50_R2plus1DMotionSegNet_model.pth":
#         model = torch.nn.DataParallel(dropout_0_50_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == "dropout_0_25_R2plus1DMotionSegNet_model.pth":
#         model = torch.nn.DataParallel(dropout_0_25_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == "dropout_0_10_R2plus1DMotionSegNet_model.pth":
#         model = torch.nn.DataParallel(dropout_0_10_R2plus1D_18_MotionNet(), device_ids = [1, 0])
    
#     # dropout v2 models
#     if model_name == 'dropout_v2_0_00_R2plus1DMotionSegNet_model.pth':
#         model = torch.nn.DataParallel(dropout_v2_0_00_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == 'dropout_v2_0_10_R2plus1DMotionSegNet_model.pth':
#         model = torch.nn.DataParallel(dropout_v2_0_10_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == 'dropout_v2_0_25_R2plus1DMotionSegNet_model.pth':
#         model = torch.nn.DataParallel(dropout_v2_0_25_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == 'dropout_v2_0_50_R2plus1DMotionSegNet_model.pth':
#         model = torch.nn.DataParallel(dropout_v2_0_50_R2plus1D_18_MotionNet(), device_ids = [1, 0])
#     if model_name == 'dropout_v2_0_75_R2plus1DMotionSegNet_model.pth':
#         model = torch.nn.DataParallel(dropout_v2_0_75_R2plus1D_18_MotionNet(), device_ids = [1, 0])
        
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
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_save_path)["model"])
    logfile.write(f'{model_name} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.\n')
    model.eval();
    
    loaded_in_models.append((model_name, model))



NUM_VIDS_TO_SEGMENT = len(test_dataset)      # max is len(test_dataset)
# NUM_VIDS_TO_SEGMENT = 1

all_seg_out_strains = [[] for i in range(len(loaded_in_models))]      # len is number of models
all_ground_truth_strains = []
vid_ind_ground_truth_strain_positive = []                             # note strange vidoes where ground truth strain val may be positive
vid_ind_seg_out_strain_positive = []                                  # notes trange videos where seg out strain val may be positive

for test_pat_index in range(NUM_VIDS_TO_SEGMENT):
    try:
        # test_pat_index = 0  # first video from test dataset
        logfile.write(f'{"~" * 64}')
        logfile.write(f'\n\nvideo number: {test_pat_index}\n\n')
        video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]


        # get all possible start points (for 32 frame clips)
        possible_starts = get_all_possible_start_points(ed_index, es_index, video.shape[1], clip_length=32)

        # segment clips by passing thru model, not consecutive 32 frame clips

        # segment using all models
        all_segmentation_outputs = []
        all_motion_outputs = []

        # for each model, segment the clips
        for name, model in loaded_in_models:

            segmentation_outputs = np.empty(shape=(1, 2, 32, 112, 112))
            motion_outputs = np.empty(shape=(1, 4, 32, 112, 112))
            for start in possible_starts:
                one_clip = np.expand_dims(video[:, start: start + 32], 0)
                segmentation_output, motion_output = model(torch.Tensor(one_clip))
                segmentation_outputs = np.concatenate([segmentation_outputs, segmentation_output.cpu().detach().numpy()])
                motion_outputs = np.concatenate([motion_outputs, motion_output.cpu().detach().numpy()])
            segmentation_outputs = segmentation_outputs[1:]
            motion_outputs = motion_outputs[1:]

            # save 
            all_segmentation_outputs.append(segmentation_outputs)
            all_motion_outputs.append(motion_outputs)

        # get "strain" vals
        # ground truth vals for current video
        y = groundTruthBoundaries(ed_label, es_label)
        ground_truth_strain = strain_value(np.count_nonzero(y[1][0] == 255), np.count_nonzero(y[1][1] == 255))
        all_ground_truth_strains.append(ground_truth_strain)
        # check if ground truth strain val is positive (shouldn't be because we are going from ED -> ES, so it should always be negative)
        if ground_truth_strain > 0.0:
            vid_ind_ground_truth_strain_positive.append(test_pat_index)

        # model vals for current video
        for i in range(len(loaded_in_models)):
            x = fromSegOutToEDESLVMasksAndBorders(all_segmentation_outputs, es_index, ed_index, model_num = i)
            logfile.write(f'{model_names[i]}\n')

            seg_out_strain = strain_value(np.count_nonzero(x[1][0] == 255), np.count_nonzero(x[1][1] == 255))
            all_seg_out_strains[i].append(seg_out_strain)
            # check if seg out strain val is positive (shouldn't be because we are going from ED -> ES, so it should always be negative)
            if seg_out_strain > 0.0:
                vid_ind_seg_out_strain_positive.append(test_pat_index)

            logfile.write(f'seg_out_strain: {seg_out_strain * 100:.4f}%\nground_truth_strain: {ground_truth_strain * 100:.4f}%\n')
            logfile.write(f'RMSE: {rmse(seg_out_strain, ground_truth_strain)}\n\n')
            
    except Exception as e:
        logfile.write(f'Failed with message: {e}\n')


# write some more stats on "strain values"
groundtruth_mean_strain = np.mean(all_ground_truth_strains)
groundtruth_median_strain = np.median(all_ground_truth_strains)
groundtruth_min_strain = np.min(all_ground_truth_strains)
groundtruth_max_strain = np.max(all_ground_truth_strains)
logfile.write(f'\nGROUND TRUTH Strain Vals:\n\n')
logfile.write(f'Mean: {groundtruth_mean_strain * 100:.4f}%\n')
logfile.write(f'Median: {groundtruth_median_strain * 100:.4f}%\n')
logfile.write(f'Min: {groundtruth_min_strain * 100:.4f}%\n')
logfile.write(f'Max: {groundtruth_max_strain * 100:.4f}%\n')

logfile.write('\nDIFFERENT MODELS Strain Vals:\n\n')
for i in range(len(loaded_in_models)):
    segout_mean_strain = np.mean(all_seg_out_strains[i])
    segout_median_strain = np.median(all_seg_out_strains[i])
    segout_min_strain = np.min(all_seg_out_strains[i])
    segout_max_strain = np.max(all_seg_out_strains[i])
    
    logfile.write(f'{loaded_in_models[i][0]}\n')
    logfile.write(f'Mean: {segout_mean_strain * 100:.4f}%\n')
    logfile.write(f'Median: {segout_median_strain * 100:.4f}%\n')
    logfile.write(f'Min: {segout_min_strain * 100:.4f}%\n')
    logfile.write(f'Max: {segout_max_strain * 100:.4f}%\n')

logfile.write(f'\nScript took {(toc() - start_time) // 60}m and {(toc() - start_time) % 60}s\n')

logfile.write('\nVideo indeces where strain was positive:\n')
logfile.write(f'Ground truth: {vid_ind_ground_truth_strain_positive}\n\n')
logfile.write(f'Seg out: {vid_ind_seg_out_strain_positive}\n')


logfile.close()