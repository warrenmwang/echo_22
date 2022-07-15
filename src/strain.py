import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from src.visualization_utils import categorical_dice



# taken from stough's dip vis_utils.py
def vis_pair(I, J, figsize = (8,3), shared = True, 
             first_title = 'Original', second_title = 'New',
             show_ticks = True, **kwargs):
    '''
    vis_pair(I, J, figsize = (8,3), shared = True, first_title = 'Original', second_title = 'New'):
    produce a plot of images I and J together. By default takes care of sharing axes to provide
    a little 1x2 plot without all the coding.
    '''
    f, ax = plt.subplots(1,2, figsize=figsize, sharex = shared, sharey = shared)
    ax[0].imshow(I, **kwargs)
    ax[0].set_title(first_title)
    ax[1].imshow(J, **kwargs)
    ax[1].set_title(second_title)
    
    if not show_ticks:
        [a.axes.get_xaxis().set_visible(False) for a in ax];
        [a.axes.get_yaxis().set_visible(False) for a in ax];
    
    plt.tight_layout()
    
def vis_pair_scatter(x1, y1, x2, y2, figsize = (8,3), shared = True, 
             first_title = 'Original', second_title = 'New',
             show_ticks = True, **kwargs):
    '''
    copy of vis_pair but for points on a scatter plot    
    y axis is inverted as well
    '''
    f, ax = plt.subplots(1,2, figsize=figsize, sharex = shared, sharey = shared)
    ax[0].scatter(x1, y1, **kwargs)
    ax[0].set_title(first_title)
    # ax[0].invert_yaxis()
    ax[1].scatter(x2, y2, **kwargs)
    ax[1].set_title(second_title)
    # ax[1].invert_yaxis()
    
    if not show_ticks:
        [a.axes.get_xaxis().set_visible(False) for a in ax];
        [a.axes.get_yaxis().set_visible(False) for a in ax];
    
    plt.tight_layout()

def vis_single(I, title='title', **kwargs):
    f,ax = plt.subplots(1,1,figsize=(8,3))
    ax.set_title(title)
    ax.imshow(I, **kwargs)



def rmse(x, y):
    ''' return root mean square error difference between the two values passed in'''
    return np.sqrt((x - y) ** 2)

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

def give_boundary(x):
    '''
    input: 
        x (112, 112) one-hot encoded lv mask/segmentation
        unique values (0,1)
        has to be numpy ndarray on cpu mem
    output: 
        y (112, 112) black and white picture of boundary of lv
        unique vals (0,1)
    '''
    foo = np.uint8(x * 255)
    ret, thresh = cv.threshold(foo, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros((112,112))
    cv.drawContours(blank, contours, -1, (255,255,255), 1)
    
    # return boundary image with unique values of (0,1)
    blank = blank / 255
    
    return blank

def boundaries_to_strain(before, after):
    '''
    input:
        before (112, 112) boundary of lv
        after (112, 112) boundary of lv
        expect unique values of (0,1)
        
    output: 
        y - floating point number representing strain value, left as a decimal, NOT multiplied by 100
    '''
    # # make sure values in image are only 0 and 1
    # check_unique_vals = np.unique(before)
    # if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
    #     before = before / 255
    # check_unique_vals = np.unique(after)
    # if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
    #     after = after / 255
        
    # cut basal plane out
    before = cut_basal_plane_out(before)
    after = cut_basal_plane_out(after)
    
    # count lv pixels
    l_0 = np.count_nonzero(before == 1)
    l_i = np.count_nonzero(after == 1)
    
    return strain_value(l_0, l_i)
    
def images_to_strain(ed_frame, es_frame):
    '''
    input:
        ed_frame (112, 112)
        es_frame (112, 112)
        
        expect vals to be one-hot encoded (1's for lv, 0's for not lv)
    output:
        x - floating point number
        strain value (some decimal, should be negative)
    '''
    # get boundaries, then cut basal plane out
    ed_bound = cut_basal_plane_out(give_boundary(ed_frame))
    es_bound = cut_basal_plane_out(give_boundary(es_frame))
    # compute strain and return
    return boundaries_to_strain(ed_bound, es_bound)

def blot_out_given_rect(rectangle_corners, I, replace_val = 0):
    '''
    input:
        rectangle_corners - shape (2,2)
        I - shape (112, 112) 
        
    output:
        I_copy -- deepcopy of original image I with the defined rectangle section given zeroed out in the image
    '''
    
     # i = row, j = col, we're thinking in ij/row col instead of xy
    if rectangle_corners[0][0] < rectangle_corners[1][0]:
        i_start = int(rectangle_corners[0][0])
        i_end = int(rectangle_corners[1][0])
    else:
        i_end = int(rectangle_corners[0][0])
        i_start = int(rectangle_corners[1][0])
        
    if rectangle_corners[1][1] < rectangle_corners[0][1]:
        j_start = int(rectangle_corners[1][1])
        j_end = int(rectangle_corners[0][1])
    else:
        j_end = int(rectangle_corners[1][1])
        j_start = int(rectangle_corners[0][1])
        
    # make copy and alter then return
    import copy 
    I_copy = copy.deepcopy(I)
    
    # zero out everything at and below the highest of the two index points (i_start)
    I_copy[i_start : I.shape[0], 0 : I.shape[1]] = replace_val
    
    return I_copy

def blank_with_rect(rectangle_corners, I, replace_val=127):
    '''
    input:
        rectangle_corners - shape (2, 2)
    output:
        empty image with the rectangle as defined blotted out with values of 127 -- shape (112, 112)
    '''
    I_rect = np.zeros(I.shape)
    I_rect = blot_out_given_rect(rectangle_corners, I_rect, replace_val=replace_val)
    return I_rect


def plot_rectangle(rectangle_corners, I=None, show_applied=False):
    '''
    input:
        rectangle_corners - shape (2,2)
        Well, technically rectangle corners could even have shape of (4, 2)
    
    plot a rectangle based on the coords given (in format of the skimage functions that get the corner pixels coords)
    also if an image is given, plot the rectangle on top of it
    '''
   
    # plot the rectangle
    I_rect = blank_with_rect(rectangle_corners, I)
    
    # plot the original image if given
    plt.figure(figsize=(10,5))
    if I is not None:
        # assuming I has unique values (0,1)
        plt.imshow(I * 255, cmap='gray', zorder=1)
        plt.imshow(I_rect, cmap='gray', zorder=2, alpha=0.5)
        # plt.colorbar()
    else:
        plt.imshow(I_rect, cmap='gray')
    
    plt.show()
    
    # show another figure of using the shown rectangle to zero out original image if asked
    if show_applied:
        plt.figure(figsize=(10,5))
        plt.imshow(blot_out_given_rect(rectangle_corners, I), cmap='gray')
        plt.show()
        
        
def vizualize_corner_pixels_detected(I, show_plot=True, return_vars=False):
    '''
    input: I (112, 112), unique vals of 0,1
    output (if requested):
        coords - numpy ndarray
        coords_subpix - nump ndarray
            shapes of 2 objects above may vary...
    
    Prints the matplotlib figure with the corner pixels detected attached to the image
    
    '''
    from skimage.feature import corner_harris, corner_subpix, corner_peaks

    # make sure values in image are only 0 and 1
    check_unique_vals = np.unique(I)
    if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
        I = I / 255
    elif check_unique_vals[0] == 0 and check_unique_vals[1] != 1:
        print('incorrect values in Image')
        return

    coords = corner_peaks(corner_harris(I), min_distance=5, threshold_rel=0.02)
    coords_subpix = corner_subpix(I, coords, window_size=13)

    if show_plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(I, cmap=plt.cm.gray)
        ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
                linestyle='None', markersize=6)
        ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
        plt.show()
    
    # return quantities if requested
    if return_vars:
        return coords, coords_subpix
    
    
def get_rectangle_corners(I):
    '''
    input:
        I - shape (112, 112)
    output:
        rectange_corners - shape (2,2)
    '''
    from skimage.feature import corner_harris, corner_subpix, corner_peaks

    # get corner pixels to form rectangle from
    coords = corner_peaks(corner_harris(I), min_distance=5, threshold_rel=0.02)    
    
    # descending order sort all possible values along axis=0 for sorting by i / row index
    coords[::-1].sort(axis=0)
    
    rectangle_corners = coords[0:2]

    return rectangle_corners


def cut_basal_plane_out(I):
    '''
    input: I (112, 112) expected to have unique values of (0,1)...1 for lv, 0 for not lv
    output: I_new (112, 112) with unique values of (0,1)... 1 for lv, 0 for not lv...this has the basal plane removed
    
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_corner.html#sphx-glr-auto-examples-features-detection-plot-corner-py
    
    When choosing which 2 points to create a rough rectangle to zero out the bottom mitral valve / basal plane boundary
    we will go with this ordering of points to use:
    1. Both points from corner_subpix
    2. One point from corner_subpix, one point from corner_peaks
    3. Both points from corner_peaks
    
    If we can't get two points at the area of the basal plane from any of the three situations described above, we are out of luck. 
    We will have to do something else, but this won't work then. 
    '''  
    # make sure values in image are only 0 and 1
    check_unique_vals = np.unique(I)
    if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
        I = I / 255
    elif check_unique_vals[0] == 0 and check_unique_vals[1] != 1:
        print('incorrect values in Image')
        return
    
    rectangle_corners = get_rectangle_corners(I)
    return blot_out_given_rect(rectangle_corners, I)


def get_seg_and_warp_data(model, test_dataset, test_pat_index):
    '''
    input: 
         model, dataset, and video index in dataset
    output:
        segmentation and motion tracking info on a specified video
        delta_ed_es = index difference between the ed and es clip
        returned seg/mot information is for a 32 frame clip where ed is at index 0
        clip_index - index of clip 
    '''
    ########################### Helper functions ###########################
    # goes thru a video and annotates where we can start clips given video length, clip length, etc.
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
        
    def generate_2dmotion_field_PLAY(x, offset):
        # Qin's code for joint_motion_seg learning works fine on our purpose too
        # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
        x_shape = x.shape
        # print(f'x_shape: {x_shape}')

        grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)

        # this should just be moving the vars to gpu mem and doing some data type conversion to some
        # floating point precision
        grid_w = grid_w.cuda().float()
        grid_h = grid_h.cuda().float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        # OLD 
        # offset_h, offset_w = torch.split(offset, 1, 1)
        # NEW , TESTING
        offset_h, offset_w = torch.split(offset, 1)

        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        offsets = torch.stack((offset_h, offset_w), 3)
        return offsets

    def categorical_dice(prediction, truth, k, epsilon=1e-5):
        """
            Compute the dice overlap between the predicted labels and truth
            Not a loss
        """
        # Dice overlap metric for label value k
        A = (prediction == k)
        B = (truth == k)
        return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)
    
    ########################################################################
    # initial grabbing of the data that we'll use
    video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
    
    # get all possible start indices for 32 frame clip with ed/es in right order
    possible_starts = get_all_possible_start_points(ed_index, es_index, video.shape[1], clip_length=32)
    
    # for now, let's use the last clip from our set of all possible clips to use
    clip_index = len(possible_starts) - 1
    # print(f'clip_index: {clip_index}')

    # get the diff in frame len from ed to es
    delta_ed_es = es_index - ed_index
    
    # use model to segment frames
    segmentation_outputs = np.empty(shape=(1, 2, 32, 112, 112))
    motion_outputs = np.empty(shape=(1, 4, 32, 112, 112))
    for start in possible_starts:
        one_clip = np.expand_dims(video[:, start: start + 32], 0)
        segmentation_output, motion_output = model(torch.Tensor(one_clip))
        segmentation_outputs = np.concatenate([segmentation_outputs, segmentation_output.cpu().detach().numpy()])
        motion_outputs = np.concatenate([motion_outputs, motion_output.cpu().detach().numpy()])
    segmentation_outputs = segmentation_outputs[1:]
    motion_outputs = motion_outputs[1:]

    # grab whatever clip we want
    curr_clip_segmentations = segmentation_outputs[clip_index]
    curr_clip_motions = motion_outputs[clip_index]
    
    return curr_clip_segmentations, curr_clip_motions, delta_ed_es, clip_index, ed_label, es_label


def one_hot(I):
    '''
    input: I - shape (2, 112, 112)
    convert a raw segmentation output into a one_hot_encoded image
    '''
    import copy
    I_copy = copy.deepcopy(I)
    return np.argmax(I_copy, 0)


def get_dice(prediction, truth):
    '''
    assume both images are one_hot_encoded
    '''
    k = 1
    return categorical_dice(prediction, truth, k, epsilon=1e-5)

def generate_2dmotion_field_custom(x, offset):
        # Qin's code for joint_motion_seg learning works fine on our purpose too
        # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
        x_shape = x.shape
        # print(f'x_shape: {x_shape}')

        grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)

        # this should just be moving the vars to gpu mem and doing some data type conversion to some
        # floating point precision
        grid_w = grid_w.cuda().float()
        grid_h = grid_h.cuda().float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        # OLD 
        # offset_h, offset_w = torch.split(offset, 1, 1)
        # NEW , TESTING
        offset_h, offset_w = torch.split(offset, 1)

        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        offsets = torch.stack((offset_h, offset_w), 3)
        return offsets

def warp_forward(I, motions, delta_ed_es, clip_index, debug=False):
    '''
    input:
        I - shape (1, 2, 112, 112), not one-hot encoded, must be the raw model segmentation output
        motions - shape ()
        delta_ed_es - integer defining how many forward iterations to take (max 31)
                        we are only interested in warping to/from ED/ES
    output:
        I_1 - shape (1, 2, 112, 112), not one-hot encoded, raw ES image, if want one-hot encoded need to apply np.argmax
        
    for now, try to do things all on the cpu
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    # convert numpy ndarrays into tensor objects moved onto device
    flow_source = torch.from_numpy(I).to(device).float()
    motions = torch.from_numpy(motions).to(device).float()

    # warping FORWARD from ED -> ES
    # python range is [x, y), inclusive start and exclusive end
    for frame_index in range(31 - delta_ed_es - clip_index, 31 - clip_index + 1, 1):
        # grab forward motion
        forward_motion = motions[:2, frame_index,...]

        # # grab the ED seg out frame to warp
        # if frame_index == 0:
        #     # flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
        #     flow_source = I
        #     print(f'type(flow_source): {type(flow_source)}')
        #     # flow_source = torch.from_numpy(flow_source).to(device).float()
        # else:
        #     pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

        # convert to tensors and move to gpu with float dtype
        # forward_motion = torch.from_numpy(forward_motion).to(device).float()
        # generate motion field for forward motion
        motion_field = generate_2dmotion_field_custom(flow_source, forward_motion)
        # create frame i+1 relative to curr frame i 
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        # use i+1 frame as next loop iter's i frame
        flow_source = next_label
        
        
        # DEBUGGING
        if debug:
            tmp = flow_source.cpu().detach().numpy()
            #vis_single(one_hot(tmp[0]), title='flow_source', cmap='seismic')
            print(f'flow_source unique vals: {np.unique(tmp)}')
            print(f'flow_source.shape: {flow_source.shape}')
            # vis_pair(tmp[0][0], tmp[0][1], first_title=f'0th @ {frame_index}', second_title=f'1th @ {frame_index}', cmap='gray')
            tmp_1 = one_hot(tmp[0])
            vis_single(tmp_1, title=f'{frame_index}, white pixels: {np.count_nonzero(tmp_1 == 1)}', cmap='gray')

    flow_source = flow_source.cpu().detach().numpy()
    
    return flow_source

def warp_backward(I, motions, delta_ed_es, clip_index, debug=False):
    '''
    input:
        I - shape (1, 2, 112, 112), not one-hot encoded, must be the raw model segmentation output
        motions - shape ()
        delta_ed_es - integer defining how many forward iterations to take (max 31)
                        we are only interested in warping to/from ED/ES
    output:
        I_1 - shape (1, 2, 112, 112), not one-hot encoded, raw ES image, if want one-hot encoded need to apply np.argmax
        
    for now, try to do things all on the cpu

    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # convert numpy ndarrays into tensor objects moved onto device
    flow_source = torch.from_numpy(I).to(device).float()
    motions = torch.from_numpy(motions).to(device).float()
    
    # warping BACKWARD from ED <- ES
    for frame_index in range(31 - clip_index, 31 - delta_ed_es - clip_index - 1, -1):
        # grab backward motion
        backward_motion = motions[2:, frame_index,...]

        # # grab the ES seg out frame to start
        # if frame_index == delta_ed_es:
        #     flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
        #     flow_source = torch.from_numpy(flow_source).to(device).float()
        # else:
        #     pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

        # convert to tensors and move to gpu with float dtype
        # backward_motion = torch.from_numpy(backward_motion).to(device).float()
        # generate motion field for backward motion
        motion_field = generate_2dmotion_field_custom(flow_source, backward_motion)
        # create frame i-1 relative to curr frame i 
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        # use i-1 frame as next loop iter's i frame
        flow_source = next_label
        
        # DEBUGGING
        if debug:
            tmp = flow_source.cpu().detach().numpy()
            #vis_single(one_hot(tmp[0]), title='flow_source', cmap='seismic')
            print(f'flow_source unique vals: {np.unique(tmp)}')
            print(f'flow_source.shape: {flow_source.shape}')
            # vis_pair(tmp[0][0], tmp[0][1], first_title=f'0th @ {frame_index}', second_title=f'1th @ {frame_index}', cmap='gray')
            tmp_1 = one_hot(tmp[0])
            vis_single(tmp_1, title=f'{frame_index}, white pixels: {np.count_nonzero(tmp_1 == 1)}', cmap='gray')


    flow_source = flow_source.cpu().detach().numpy()
    
    return flow_source    

def top_bottom_index(I):
    '''
    input:
        I - shape (112, 112), unique vals (0,1)
    output:
        top, bot -- index vals of highest pixel with val 1 and lowest pixel with val 1
                    i.e. range of the heart
                    
                    top inclusive
                    bot exclusive
                        Dijkstra would like me decision ;)
    '''
    # make sure values in image are only 0 and 1
    check_unique_vals = np.unique(I)
    if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
        I = I / 255
    elif check_unique_vals[0] == 0 and check_unique_vals[1] != 1:
        print('incorrect values in Image')
        return
    
    top = 0
    bottom = 0
    delta = 0
    
    found_top = False
    found_bottom = False
    
    for row in I:
        # find the top
        if not found_top:
            if 1 in row:
                found_top = True
                top = delta
            
        # find bottom
        if not found_bottom and found_top:
            if 1 not in row:
                found_bottom = True
                bottom = delta
                
        # leave when found both
        if found_top and found_bottom:
            break
            
        # increment row index
        delta += 1
            
    return top, bottom

def get_split_points(I, N, inds):
    '''
    input: 
        I - shape (112, 112), unique vals (0,1)
        N - int
        inds - list/iterable containing top and bottom index of heart (top, bot)
    output:
        list of N points where we divide the image
    '''
    top = inds[0]
    bot = inds[1]
    
    lv_vert_len = bot - top
    deltas = lv_vert_len // N
    divide_points = [top + (deltas * i) for i in range(N)]
    divide_points.append(bot)
    
    return divide_points

######################

def get_warped_ed_es_frames(test_pat_index, test_dataset, model):
    '''
    INPUT: 
    test_pat_index and test_dataset to be used like the following:
        video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
    model
        model to be used, one model at a time
    clip_index
        the clip we should use, determines the position of the ED and ES frame in the 32 frame clip
        the difference in frames from ED to ES is always the same

    OUTPUT:
    es_created_from_warping_ed, ed_created_from_warping_es
        both with shape: (1, 2, 112, 112)
        you should double check the shape.
    and other vars 
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ########################### Helper functions ###########################
    # goes thru a video and annotates where we can start clips given video length, clip length, etc.
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
        
    def generate_2dmotion_field_PLAY(x, offset):
        # Qin's code for joint_motion_seg learning works fine on our purpose too
        # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
        x_shape = x.shape
        # print(f'x_shape: {x_shape}')

        grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)

        # this should just be moving the vars to gpu mem and doing some data type conversion to some
        # floating point precision
        grid_w = grid_w.cuda().float()
        grid_h = grid_h.cuda().float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        # OLD 
        # offset_h, offset_w = torch.split(offset, 1, 1)
        # NEW , TESTING
        offset_h, offset_w = torch.split(offset, 1)

        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        offsets = torch.stack((offset_h, offset_w), 3)
        return offsets

    def categorical_dice(prediction, truth, k, epsilon=1e-5):
        """
            Compute the dice overlap between the predicted labels and truth
            Not a loss
        """
        # Dice overlap metric for label value k
        A = (prediction == k)
        B = (truth == k)
        return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)
    
    ########################################################################
    # initial grabbing of the data that we'll use
    video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
    
    # get all possible start indices for 32 frame clip with ed/es in right order
    possible_starts = get_all_possible_start_points(ed_index, es_index, video.shape[1], clip_length=32)
    
    # for now, let's use the last clip from our set of all possible clips to use
    clip_index = len(possible_starts) - 1
    # print(f'clip_index: {clip_index}')

    # get the diff in frame len from ed to es
    delta_ed_es = es_index - ed_index
    
    # use model to segment frames
    segmentation_outputs = np.empty(shape=(1, 2, 32, 112, 112))
    motion_outputs = np.empty(shape=(1, 4, 32, 112, 112))
    for start in possible_starts:
        one_clip = np.expand_dims(video[:, start: start + 32], 0)
        segmentation_output, motion_output = model(torch.Tensor(one_clip))
        segmentation_outputs = np.concatenate([segmentation_outputs, segmentation_output.cpu().detach().numpy()])
        motion_outputs = np.concatenate([motion_outputs, motion_output.cpu().detach().numpy()])
    segmentation_outputs = segmentation_outputs[1:]
    motion_outputs = motion_outputs[1:]

    # grab whatever clip we want
    curr_clip_segmentations = segmentation_outputs[clip_index]
    curr_clip_motions = motion_outputs[clip_index]
    
    # print(f'curr_clip_segmentations: {curr_clip_segmentations.shape}')
    # print(f'curr_clip_motions: {curr_clip_motions.shape}')
    
    ######################## Warp from ED -> ES and ED <- ES ########################
    
    # remember, we will want to continuously warp the previous frame that has been warped forward in time
    # only the first frame that we start with will be the actual seg out frame
    flow_source = None

    # warping FORWARD from ED -> ES
    # python range is [x, y), inclusive start and exclusive end
    for frame_index in range(31 - delta_ed_es - clip_index, 31 - clip_index + 1, 1):
        # grab forward motion
        forward_motion = curr_clip_motions[:2, frame_index,...]

        # grab the ED seg out frame to warp
        if frame_index == 0:
            flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
            flow_source = torch.from_numpy(flow_source).to(device).float()
        else:
            pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

        # convert to tensors and move to gpu with float dtype
        forward_motion = torch.from_numpy(forward_motion).to(device).float()
        # generate motion field for forward motion
        motion_field = generate_2dmotion_field_PLAY(flow_source, forward_motion)
        # create frame i+1 relative to curr frame i 
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        # use i+1 frame as next loop iter's i frame
        flow_source = next_label

    es_created_from_warping_ed = flow_source.cpu().detach().numpy()

    # warping BACKWARD from ED <- ES
    for frame_index in range(31 - clip_index, 31 - delta_ed_es - clip_index - 1, -1):
        # grab backward motion
        backward_motion = curr_clip_motions[2:, frame_index,...]

        # grab the ES seg out frame to start
        if frame_index == delta_ed_es:
            flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
            flow_source = torch.from_numpy(flow_source).to(device).float()
        else:
            pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

        # convert to tensors and move to gpu with float dtype
        backward_motion = torch.from_numpy(backward_motion).to(device).float()
        # generate motion field for backward motion
        motion_field = generate_2dmotion_field_PLAY(flow_source, backward_motion)
        # create frame i-1 relative to curr frame i 
        next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
        # use i-1 frame as next loop iter's i frame
        flow_source = next_label

    ed_created_from_warping_es = flow_source.cpu().detach().numpy()
    
    ######################## ######################## ########################
    
    ed_created_from_warping_es = np.argmax(ed_created_from_warping_es, 1)[0]

    es_created_from_warping_ed = np.argmax(es_created_from_warping_ed, 1)[0]

    
    return es_created_from_warping_ed, ed_created_from_warping_es