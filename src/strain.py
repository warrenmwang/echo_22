import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from src.visualization_utils import categorical_dice
# for get2dpucks
from skimage.transform import (resize, 
                               rescale)
from skimage.segmentation import find_boundaries
import copy


#########################################################################################
# VISUALIZATION FUNCS

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

def vis_one_point_set(ps):
    f, ax = plt.subplots(1,1,figsize=(4,3), sharex=True, sharey=True)
    ax.scatter(ps[:, 1], ps[:,  0], marker='.', color='b')
    ax.invert_yaxis()
    
def vis_two_point_sets(ps_1, ps_2):
    f, ax = plt.subplots(1,2,figsize=(10,7), sharex=True, sharey=True)
    ax[0].scatter(ps_1[:, 1], ps_1[:,  0], marker='.', color='b')
    ax[0].invert_yaxis()
    
    ax[1].scatter(ps_2[:, 1], ps_2[:,  0], marker='.', color='b')
    ax[1].invert_yaxis()
    
    
def vis_three_point_sets(I_regional_point_sets):
    # this assumes we stored in i,j
    fig, ax = plt.subplots(1,3, figsize=(10,7), sharex = True, sharey = True)
    ax[0].scatter(I_regional_point_sets[0][:, 1], I_regional_point_sets[0][:, 0], marker='.', color='b')
    ax[0].invert_yaxis()

    ax[1].scatter(I_regional_point_sets[1][:, 1], I_regional_point_sets[1][:, 0], marker='.', color='b')
    ax[1].invert_yaxis()

    ax[2].scatter(I_regional_point_sets[2][:, 1], I_regional_point_sets[2][:, 0], marker='.', color='b')
    ax[2].invert_yaxis()
    
# def vis_double_three_point_sets(I_regional_point_sets, new_regional_point_sets, title='title'):
#     '''
#     I_regional_point_sets -- stored in (i,j)
    
#     new_point_set -- stored in (x, y)
    
#     :-) im sorry
#     '''
#     fig, ax = plt.subplots(1,3, figsize=(10,7), sharex = True, sharey = True)
#     ax[0].scatter(I_regional_point_sets[0][:, 1], I_regional_point_sets[0][:, 0], marker='.', color='b', zorder=1)
#     ax[0].scatter(new_regional_point_sets[0][:, 0], new_regional_point_sets[0][:, 1], marker='.', color='r', zorder=2)
#     ax[0].invert_yaxis()
#     ax[0].set_title('Apical')

#     ax[1].scatter(I_regional_point_sets[1][:, 1], I_regional_point_sets[1][:, 0], marker='.', color='b', zorder=1)
#     ax[1].scatter(new_regional_point_sets[1][:, 0], new_regional_point_sets[1][:, 1], marker='.', color='r', zorder=2)
#     ax[1].invert_yaxis()
#     ax[1].set_title('Mid')

#     ax[2].scatter(I_regional_point_sets[2][:, 1], I_regional_point_sets[2][:, 0], marker='.', color='b', zorder=1)
#     ax[2].scatter(new_regional_point_sets[2][:, 0], new_regional_point_sets[2][:, 1], marker='.', color='r', zorder=2)
#     ax[2].invert_yaxis()
#     ax[2].set_title('Basal')
    
#     fig.suptitle(title)
    
    
def view_4_vectors_and_interped_vector(vectors, new_vector):
    '''
    view 4 vectors at a unit square with a new interpolated vector (all vectors given)
    all vectors are Vector ADT
    '''
    x_tails = []
    y_tails = []
    x_mags = []
    y_mags = []

    for v in vectors:
        x_tails.append(v.tail_x)
        y_tails.append(v.tail_y)

        x_mags.append(v.mag_x)
        y_mags.append(v.mag_y)
        
    plt.figure()
    plt.quiver(x_tails, y_tails, x_mags, y_mags, color='k', linewidth=0.7)
    plt.quiver(new_vector.tail_x, new_vector.tail_y, new_vector.mag_x, new_vector.mag_y, color='b', linewidth=0.7)    

def vis_one_vec_multiple_frames(all_vectors):
    '''
    input: all_vectors - python list of Vector ADTs
    
    '''
    
    x_tails = []
    y_tails = []
    x_mags = []
    y_mags = []

    for v in all_vectors:
        x_tails.append(v.tail_x)
        y_tails.append(v.tail_y)

        x_mags.append(v.mag_x)
        y_mags.append(v.mag_y)

    # vector field
    plt.figure()
    plt.quiver(x_tails, y_tails, x_mags, y_mags, color='k', linewidth=0.7, angles='xy', scale_units='xy', units='xy', scale = 1)
    plt.show()
    
    # scatter plot, color gradient points
    plt.figure()
    plt.scatter(x_tails, y_tails, c=np.linspace(0,1,len(x_tails)))
    plt.show()
#########################################################################################

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


def give_boundary_no_basal_plane(I):
    '''
    input: I (H, W) expected to have unique values of (0,1)
    output: I_new (H, W) with unique vals of (0,1) -- with boundary pixels only and with basal plane cut out
    '''
    return cut_basal_plane_out(give_boundary(I))

#########################################################################################

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

   
# commented out bc these were doing the wrong thing, may or may not find them useful for future changes
# def top_bottom_index(I):
#     '''
#     input:
#         I - shape (112, 112), unique vals (0,1)
#     output:
#         top, bot -- index vals of highest pixel with val 1 and lowest pixel with val 1
#                     i.e. range of the heart
                    
#                     top inclusive
#                     bot exclusive
#                         Dijkstra would like me decision ;)
#     '''
#     # make sure values in image are only 0 and 1
#     check_unique_vals = np.unique(I)
#     if check_unique_vals[0] == 0 and check_unique_vals[1] == 255:
#         I = I / 255
#     elif check_unique_vals[0] == 0 and check_unique_vals[1] != 1:
#         print('incorrect values in Image')
#         return
    
#     top = 0
#     bottom = 0
#     delta = 0
    
#     found_top = False
#     found_bottom = False
    
#     for row in I:
#         # find the top
#         if not found_top:
#             if 1 in row:
#                 found_top = True
#                 top = delta
            
#         # find bottom
#         if not found_bottom and found_top:
#             if 1 not in row:
#                 found_bottom = True
#                 bottom = delta
                
#         # leave when found both
#         if found_top and found_bottom:
#             break
            
#         # increment row index
#         delta += 1
            
#     return top, bottom

# def get_split_points(I, N, inds):
#     '''
#     input: 
#         I - shape (112, 112), unique vals (0,1)
#         N - int
#         inds - list/iterable containing top and bottom index of heart (top, bot)
#     output:
#         list of N points where we divide the image
#     '''
#     top = inds[0]
#     bot = inds[1]
    
#     lv_vert_len = bot - top
#     deltas = lv_vert_len // N
#     divide_points = [top + (deltas * i) for i in range(N)]
#     divide_points.append(bot)
    
#     return divide_points

#########################################################################################

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
    model.eval()
    
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


# def get_warped_ed_es_frames(test_pat_index, test_dataset, model):
#     '''
#     INPUT: 
#     test_pat_index and test_dataset to be used like the following:
#         video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
#     model
#         model to be used, one model at a time
#     clip_index
#         the clip we should use, determines the position of the ED and ES frame in the 32 frame clip
#         the difference in frames from ED to ES is always the same

#     OUTPUT:
#     es_created_from_warping_ed, ed_created_from_warping_es
#         both with shape: (1, 2, 112, 112)
#         you should double check the shape.
#     and other vars 
#     '''
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     ########################### Helper functions ###########################
#     # goes thru a video and annotates where we can start clips given video length, clip length, etc.
#     def get_all_possible_start_points(ed_index, es_index, video_length, clip_length):
#         assert es_index - ed_index > 0, "not a ED to ES clip pair"
#         possible_shift = clip_length - (es_index - ed_index)
#         allowed_right = video_length - es_index
#         if allowed_right < possible_shift:
#             return np.arange(ed_index - possible_shift + 1, video_length - clip_length + 1)
#         if possible_shift < 0:
#             return np.array([ed_index])
#         elif ed_index < possible_shift:
#             return np.arange(ed_index + 1)
#         else:
#             return np.arange(ed_index - possible_shift + 1, ed_index + 1)
        
#     def generate_2dmotion_field_PLAY(x, offset):
#         # Qin's code for joint_motion_seg learning works fine on our purpose too
#         # Same idea https://discuss.pytorch.org/t/warp-video-frame-from-optical-flow/6013/5
#         x_shape = x.shape
#         # print(f'x_shape: {x_shape}')

#         grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)

#         # this should just be moving the vars to gpu mem and doing some data type conversion to some
#         # floating point precision
#         grid_w = grid_w.cuda().float()
#         grid_h = grid_h.cuda().float()

#         grid_w = nn.Parameter(grid_w, requires_grad=False)
#         grid_h = nn.Parameter(grid_h, requires_grad=False)

#         # OLD 
#         # offset_h, offset_w = torch.split(offset, 1, 1)
#         # NEW , TESTING
#         offset_h, offset_w = torch.split(offset, 1)

#         offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
#         offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

#         offset_w = grid_w + offset_w
#         offset_h = grid_h + offset_h

#         offsets = torch.stack((offset_h, offset_w), 3)
#         return offsets

#     def categorical_dice(prediction, truth, k, epsilon=1e-5):
#         """
#             Compute the dice overlap between the predicted labels and truth
#             Not a loss
#         """
#         # Dice overlap metric for label value k
#         A = (prediction == k)
#         B = (truth == k)
#         return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)
    
#     ########################################################################
#     # initial grabbing of the data that we'll use
#     video, (filename, EF, es_clip_index, ed_clip_index, es_index, ed_index, es_frame, ed_frame, es_label, ed_label) = test_dataset[test_pat_index]
    
#     # get all possible start indices for 32 frame clip with ed/es in right order
#     possible_starts = get_all_possible_start_points(ed_index, es_index, video.shape[1], clip_length=32)
    
#     # for now, let's use the last clip from our set of all possible clips to use
#     clip_index = len(possible_starts) - 1
#     # print(f'clip_index: {clip_index}')

#     # get the diff in frame len from ed to es
#     delta_ed_es = es_index - ed_index
    
#     # use model to segment frames
#     segmentation_outputs = np.empty(shape=(1, 2, 32, 112, 112))
#     motion_outputs = np.empty(shape=(1, 4, 32, 112, 112))
#     for start in possible_starts:
#         one_clip = np.expand_dims(video[:, start: start + 32], 0)
#         segmentation_output, motion_output = model(torch.Tensor(one_clip))
#         segmentation_outputs = np.concatenate([segmentation_outputs, segmentation_output.cpu().detach().numpy()])
#         motion_outputs = np.concatenate([motion_outputs, motion_output.cpu().detach().numpy()])
#     segmentation_outputs = segmentation_outputs[1:]
#     motion_outputs = motion_outputs[1:]

#     # grab whatever clip we want
#     curr_clip_segmentations = segmentation_outputs[clip_index]
#     curr_clip_motions = motion_outputs[clip_index]
    
#     # print(f'curr_clip_segmentations: {curr_clip_segmentations.shape}')
#     # print(f'curr_clip_motions: {curr_clip_motions.shape}')
    
#     ######################## Warp from ED -> ES and ED <- ES ########################
    
#     # remember, we will want to continuously warp the previous frame that has been warped forward in time
#     # only the first frame that we start with will be the actual seg out frame
#     flow_source = None

#     # warping FORWARD from ED -> ES
#     # python range is [x, y), inclusive start and exclusive end
#     for frame_index in range(31 - delta_ed_es - clip_index, 31 - clip_index + 1, 1):
#         # grab forward motion
#         forward_motion = curr_clip_motions[:2, frame_index,...]

#         # grab the ED seg out frame to warp
#         if frame_index == 0:
#             flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
#             flow_source = torch.from_numpy(flow_source).to(device).float()
#         else:
#             pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

#         # convert to tensors and move to gpu with float dtype
#         forward_motion = torch.from_numpy(forward_motion).to(device).float()
#         # generate motion field for forward motion
#         motion_field = generate_2dmotion_field_PLAY(flow_source, forward_motion)
#         # create frame i+1 relative to curr frame i 
#         next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
#         # use i+1 frame as next loop iter's i frame
#         flow_source = next_label

#     es_created_from_warping_ed = flow_source.cpu().detach().numpy()

#     # warping BACKWARD from ED <- ES
#     for frame_index in range(31 - clip_index, 31 - delta_ed_es - clip_index - 1, -1):
#         # grab backward motion
#         backward_motion = curr_clip_motions[2:, frame_index,...]

#         # grab the ES seg out frame to start
#         if frame_index == delta_ed_es:
#             flow_source = np.array([curr_clip_segmentations[:, frame_index, ...]])
#             flow_source = torch.from_numpy(flow_source).to(device).float()
#         else:
#             pass # use previous next_label as flow_source, should be redefined at end of previous loop iter

#         # convert to tensors and move to gpu with float dtype
#         backward_motion = torch.from_numpy(backward_motion).to(device).float()
#         # generate motion field for backward motion
#         motion_field = generate_2dmotion_field_PLAY(flow_source, backward_motion)
#         # create frame i-1 relative to curr frame i 
#         next_label = F.grid_sample(flow_source, motion_field, align_corners=False, mode="bilinear", padding_mode='border')
#         # use i-1 frame as next loop iter's i frame
#         flow_source = next_label

#     ed_created_from_warping_es = flow_source.cpu().detach().numpy()
    
#     ######################## ######################## ########################
    
#     ed_created_from_warping_es = np.argmax(ed_created_from_warping_es, 1)[0]

#     es_created_from_warping_ed = np.argmax(es_created_from_warping_ed, 1)[0]

    
#     return es_created_from_warping_ed, ed_created_from_warping_es


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

#########################################################################################
# FOR SLICING A SEGMENTATION INTO THE BOUNDARY POINT SETS

# let's try to use 9 pucks, if we divide into N=3 sections, we define each section to be 3 pucks
def get2dPuckEndpoints(abin, apix, npucks=9):
    '''
    Originally:
        get2dPucks(abin, apix): Return the linear extent of the binary structure,
        as well as a sequence of radii about that extent.
    Now just used to return the endpoints of the pucks representing the endpoints of the individual
    radii parallel to the minor axis
    '''
    
    all_puck_endpoints = []
    
    # Empty bin?
    if ~np.any(abin):
        return 1.0, np.zeros((npucks,))
    
    x,y = np.where(abin>0)
    X = np.stack([x,y]) # Coords of all non-zero pixels., 2 x N
    if X.shape[1] < 1: # no pixels, doesn't seem to be a problem usually. 
        return (0.0, np.zeros((npucks,)))
    # Scale dimensions
    X = np.multiply(X, np.array(apix)[:, None]) # Need a broadcastable shape, here 2 x 1
    try:
        val, vec = np.linalg.eig(np.cov(X, rowvar=True))
    except:
        return (0.0, np.zeros((npucks,)))
    
    # Make sure we're in decreasing order of importance.
    eigorder = np.argsort(val)[-1::-1]
    vec = vec[:, eigorder]
    val = val[eigorder]
    
    # Negate the eigenvectors for consistency. Let's say y should be positive eig0,
    # and x should be positive for eig1. I'm not sure if that's what I'm doing here,
    # but just trying to be consistent.
    if vec[0,0] < 0:
        vec[:,0] = -1.0*vec[:,0]
    if vec[1,1] < 0:
        vec[:,1] = -1.0*vec[:,1]
    
    mu = np.expand_dims(np.mean(X, axis=1), axis=1)
    
    # Now mu is 2 x 1 mean pixel coord 
    # val is eigenvalues, vec is 2 x 2 right eigvectors (by column), all in matrix ij format
    
    # Use the boundary to get the radii.
    # Project the boundary pixel coords into the eigenspace.
    B = find_boundaries(abin, mode='thick')
    Xb = np.stack(np.where(B))
    Xb = np.multiply(Xb, np.array(apix)[:, None]) # space coords again.
    proj = np.dot((Xb-mu).T,vec) 
    # proj is M x 2, the projections onto 0 and 1 eigs of the M boundary coords.
    
    # Now get min max in the first principal direction. That's L! Just L[0] here.
    L_min, L_max = np.min(proj, axis=0), np.max(proj, axis=0)
    L = L_max - L_min
    
    # Partition along the principal axis. The secondary axis represents the radii.
    L_partition = np.linspace(L_min[0], L_max[0], npucks+1)
    
    R = []
    A = np.copy(proj)
    for i in range(len(L_partition)-1):
        # Select those boundary points whose projection on the major axis
        # is within the thresholds. 
        which = np.logical_and(A[:,0] >= L_partition[i],
                               A[:,0] < L_partition[i+1])
        # here which could be empty, if there are multiple components to the binary,
        # which will happen without cleaning for the largest connected component and 
        # such. r will be nan, here I replace with zero.
        # In fact, this math really only works well with nice convex objects.
        if len(which) == 0:
            r = 0
        else:
            r = np.median(np.abs(A[:,1][which]))
        R.append(r)
    
    # Some visualization code I didn't know where else to put!
    # B is still in image coords, while mu and the vec and L's are in mm? Use extent.
    # extent = (-0.5, apix[1]*B.shape[1]-0.5, -0.5, apix[0]*B.shape[0]-0.5)# (left, right, bottom, top)

    # This got me pretty confused. The issue is that if apix is something other than (1,1), then 
    # B needs to be scaled accordingly. 
    # If apix is significantly less than 1,1, then the 0 order and no anti-aliasing could
    # leave little of the boundary left. Though it would only affect the vis, as the calculation
    # above scaled the boundary points to double, instead of this which returns pixels.
    abin_scaled = rescale(abin, apix, order=0, 
                          preserve_range=True, 
                          anti_aliasing=False, 
                          multichannel=False)
    Bscaled = find_boundaries(abin_scaled, mode='thick')


    # Plot the mean and principal projections. But plot needs xy (euclid) instead of ij (matrix)!!!
    # Stupid, keeping the sliced out dimension with None here.
    pca0 = np.array([mu + L_min[0]*vec[:,0, None], mu + L_max[0]*vec[:,0, None]])
    pca1 = np.array([mu + L_min[1]*vec[:,1, None], mu + L_max[1]*vec[:,1, None]])

    # Notice the x and y coord reversed. 

    # these are only the end points of the major and minor axes I think ? 
    # plt.scatter(x=mu[1], y=mu[0], s=30, marker='*') # what is this one? ?? -- hard to tell, no visual difference?


    # this plots all of the points of the individual lines of the npucks 
    for i in range(len(L_partition)-1):

        extent = (L_partition[i]+L_partition[i+1])/2
        points = np.array([mu + extent*vec[:,0, None] - R[i]*vec[:,1, None], # negative radius
                           mu + extent*vec[:,0, None] + R[i]*vec[:,1, None]]) # positive radius

        all_puck_endpoints.append(np.transpose(points))
    
    # return L[0], np.array(R), np.array(all_puck_endpoints)
    return np.array(all_puck_endpoints)


def is_point_above_below_line(p1, p2, p3):
    '''
    input: p1, p2, p3 -- iterables containing (x,y)
        p1 and p2 represent line and p3 is point to be checked
        note that this is supposed to be in euclidean space (x,y) 
        if points are in (i,j) please convert them first into (x,y)
        x = j, y = i
    output:
        +1 for above line
        -1 for below line
        0 for on the line
    '''
    a = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0])
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0
    
def ij_to_xy(p):
    '''
    input: p - (i,j)
    output: p - (x,y)
    x = j, y = i
    '''
    return np.array([p[1], p[0]])

def image_to_regional_point_sets(I, N=3):
    '''
    input:
        I - image (112, 112) unique vals of (0,1) -- should be the full segmentation!!!
        N - number of regions to slice
            note if this should be changed get2dPuckEndpoints npucks param needs to be divisible by N
    output:
        I_regional_point_sets - (N, X, 2) - stored in (i,j) format
            X is not the same for different index at axis=0
            .shape on ths object will show (N, )
    '''
    
    radiiEndpoints_I = get2dPuckEndpoints((I == 1).astype('int'), (1.0, 1.0), npucks=3 * N)
    # after get points from segmentation, now transform into boundary points with mitral valve removed
    I = give_boundary_no_basal_plane(I)
    
    # get the indeces of where we'll slice
    indeces_for_slice_points = []
    for i in range(len(radiiEndpoints_I)-1, 0, -N):
        indeces_for_slice_points.append(i)
    indeces_for_slice_points.sort()

    # get the actual points of where we'll slice
    slice_points = []
    for i in indeces_for_slice_points:
        slice_points.append(radiiEndpoints_I[i, 0, ...])
    slice_points = np.array(slice_points)

    # reorganize our points to be stored as (i,j)
    divide_points = [] # (i,j)
    for a in slice_points:
        divide_points.append([ [a[0][0], a[1][0]] , [a[0][1], a[1][1]] ])
        
    # convert input image into points
    I_point_set = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i][j] == 1:
                I_point_set.append([i,j])

                
    # split input image point set into the regional point sets
    I_regional_point_sets = [0 for i in range(N)]
    for i in range(N):
        curr_region_point_set = []

        # first region
        if i == 0:
            p1 = ij_to_xy(divide_points[i][0])
            p2 = ij_to_xy(divide_points[i][1])

            for point in I_point_set:
                p3 = ij_to_xy(point)

                if is_point_above_below_line(p1, p2, p3) == -1:
                    curr_region_point_set.append(point)


        # 2nd and + regions
        else:
            start_p1 = ij_to_xy(divide_points[i-1][0])
            start_p2 = ij_to_xy(divide_points[i-1][1])

            end_p1 = ij_to_xy(divide_points[i][0])
            end_p2 = ij_to_xy(divide_points[i][1])

            # look at the entire point set representing the entire lv
            for point in I_point_set:
                p3 = ij_to_xy(point)

                check_1 = is_point_above_below_line(start_p1, start_p2, p3)
                check_2 = is_point_above_below_line(end_p1, end_p2, p3)

                # above the first line and below the second line
                # save point in (i,j) format
                if check_1 == 1 and check_2 == -1:
                    curr_region_point_set.append(point)

        # save curr region point set
        I_regional_point_sets[i] = np.array(curr_region_point_set)

    I_regional_point_sets = np.array(I_regional_point_sets)
    
    return I_regional_point_sets

#########################################################################################
# DOING SPHERICAL BILINEAR INTERPOLATION
# ( BILINEAR INTERP ON (RHO, THETA) REPRESENTING VECTOR MAGNITUDE ) WHERE (RHO, THETA) ARE POLAR COORDS
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

class Vector:
    '''
    class for representing vectors used in warping frames using the motion tracking data
    '''
    def __init__(self, tail_x, tail_y, mag_x, mag_y):
        self.tail_x = tail_x
        self.tail_y = tail_y
        
        self.mag_x = mag_x
        self.mag_y = mag_y
        
        # for converting vector's magnitude between polar
        # and cartesian
        self.conversion_mag_rho = 0
        self.conversion_mag_theta = 0
        
        self.conversion_mag_x = 0
        self.conversion_mag_y = 0
        
    def __str__(self):
        return f'tail_x: {self.tail_x}\ntail_y: {self.tail_y}\nmag_x: {self.mag_x}\nmag_y: {self.mag_y}\n'
        
    def polar(self):
        self.conversion_mag_rho, self.conversion_mag_theta = cart2pol(self.mag_x, self.mag_y)
        
    def cart(self):
        self.conversion_mag_x, self.conversion_mag_y = pol2cart(self.conversion_mag_rho, self.conversion_mag_theta)
    
        
    def update_mag_xy_from_conversion(self):
        self.mag_x = self.conversion_mag_x
        self.mag_y = self.conversion_mag_y
        
    def update_tails_from_mags(self):
        self.tail_x += self.mag_x
        self.tail_y += self.mag_y 
        
    def clear_magnitudes(self):
        self.mag_x = 0
        self.mag_y = 0
        
    def clear_conversion_mags_xy(self):
        self.conversion_mag_x = 0
        self.conversion_mag_y = 0
        
    def clear_conversion_mags_rhotheta(self):
        self.conversion_mag_rho = 0
        self.conversion_mag_theta = 0
        

        
def vector_bilinear_interpolation(vectors, new_vector):
    '''
    input: 
        vectors - list of 4 Vectors (our class def)
            base values (x_i, y_i, delta_x_i, delta_y_i)
        new_vector - single Vector which is just a point with 0 x and y magnitude
            base values (x_i, y_i, 0, 0)
    
    output:
        new_vector - same input Vector, except with bilinearly interpolated in polar and then converted 
            back into cartesian x and y magnitudes of direction using the 4 vectors 
            were surrounding this vector at integer values, essentially making up a unit square
            around this vector
                base values (x_i, y_i, delta_x, delta_y)
    '''
    # convert magnitudes to polar
    for v in vectors:
        v.polar()
        
    # linearly interpolate top two points
    # linearly interpoalte bottom two points
    # then linearly interpolate with the two new points
    top_vectors = []
    bottom_vectors = []

    for v in vectors:
        if v.tail_y == int(new_vector.tail_y):
            bottom_vectors.append(v)
        else:
            top_vectors.append(v)
            
    # for first two linearly interps
    weight_1 = new_vector.tail_x - top_vectors[0].tail_x
    # print(f'weights: {(weight_1, weight_2)}')

    # horizontal top
    rho_1 = (weight_1 * top_vectors[1].conversion_mag_rho) + ((1.0 - weight_1) * top_vectors[0].conversion_mag_rho)
    theta_1 = (weight_1 * top_vectors[1].conversion_mag_theta) + ((1.0 - weight_1) * top_vectors[0].conversion_mag_theta)

    # horizontal bottom
    rho_2 = (weight_1 * bottom_vectors[1].conversion_mag_rho) + ((1.0 - weight_1) * bottom_vectors[0].conversion_mag_rho)
    theta_2 = (weight_1 * bottom_vectors[1].conversion_mag_theta) + ((1.0 - weight_1) * bottom_vectors[0].conversion_mag_theta)

    # final interp
    weight_2 = new_vector.tail_y - bottom_vectors[0].tail_y
    # vertial
    rho_3 = (weight_2 * rho_1) + ((1.0 - weight_2) * rho_2)
    theta_3 = (weight_2 * theta_1) + ((1.0 - weight_2) * theta_2)       
    
    new_vector.conversion_mag_rho = rho_3
    new_vector.conversion_mag_theta = theta_3
    new_vector.cart()
    
    return new_vector


##### 

# OLD warping with vector functions
# def preliminary_warp_one_vector_forward(coords_multiple_frames, curr_clip_motions):
#     '''
#     input:
#         coords_multiple_frames - (1, N, 2)
#             contains the points of a regional point set in ij
#             axis=0 , first dim at ED frame, second dim at ED+1 frame got from directly applying motion tracking info (x+delta_x_0, y+delta_y_0)
#         curr_clip_motions - (4, 32, 112, 112)
#     output:
#         an updated coords_multiple_frames - (2, N, 2)
#             with the second dim at ax=0 being the new points warped from the ED by one frame
#     '''
#     frame = 0

#     # warp forward all the current points
#     coords_1 = []
#     for _ in range(coords_multiple_frames.shape[1]):
#         pair = coords_multiple_frames[0][_]

#         i_0,j_0 = int(pair[0]), int(pair[1])

#         # forward change in y
#         delta_i = curr_clip_motions[1][frame][i_0][j_0]

#         # forward change in x
#         delta_j = curr_clip_motions[0][frame][i_0][j_0]

#         i_1, j_1 = i_0 + delta_i , j_0 + delta_j

#         coords_1.append([i_1, j_1])

#     coords_1 = np.array(coords_1)

#     coords_multiple_frames = np.insert(coords_multiple_frames, 1, coords_1, axis=0)

#     return coords_multiple_frames

# def warp_one_vector_forward_till_ES(coords_multiple_frames, ind_of_point_to_warp, delta_ed_es, curr_clip_motions):
#     # the first vector from frame 0 -> 1 (don't be confused, storing the (i,j) and motion tracking (forward x,y) that gets us from ED to ED+1
#     # we already applied that information in the preliminary warp, so this is just redundantly storing that information
#     all_new_vectors = []

#     point_0 = coords_multiple_frames[0][ind_of_point_to_warp]
#     point_0_i = int(point_0[0])
#     point_0_j = int(point_0[1])

#     mag_x = curr_clip_motions[0][0][point_0_i][point_0_j]
#     mag_y = curr_clip_motions[1][0][point_0_i][point_0_j]
#     thing = Vector(point_0_j, point_0_i, mag_x, mag_y)
#     all_new_vectors.append(copy.deepcopy(thing))
    
#     # the next vector from frame 1 -> 2 (BUT, now we need to do interp to get correct magnitudes!)

#     point_1 = coords_multiple_frames[1][ind_of_point_to_warp] # one point at frame 1, need to now warp using frame 1 vectors
#     new_vector = Vector(point_1[1], point_1[0], 0, 0)
#     all_new_vectors.append(copy.deepcopy(new_vector))
    
#     # we already warped 1 frame, so let's warp from frame=1 to delta_ed_es - 1
#     for frame in range(1, delta_ed_es - 1):

#         inted_i, inted_j = int(new_vector.tail_y), int(new_vector.tail_x)
#         surr_vec_tails = [ [inted_i, inted_j],
#                              [inted_i, inted_j+1],
#                              [inted_i+1, inted_j],
#                              [inted_i+1, inted_j+1] ]

#         surround_vectors = []

#         for _ in surr_vec_tails:
#             i,j = _[0], _[1]
#             x = j
#             y = i

#             mag_i = curr_clip_motions[1][frame][i][j]
#             mag_j = curr_clip_motions[0][frame][i][j]

#             mag_x = mag_j
#             mag_y = mag_i

#             surround_vectors.append(Vector(x, y, mag_x, mag_y))

#         new_vector = vector_bilinear_interpolation(surround_vectors, new_vector)

#         new_vector.tail_x += new_vector.conversion_mag_x
#         new_vector.tail_y += new_vector.conversion_mag_y

#         all_new_vectors.append(copy.deepcopy(new_vector))

#         new_vector.conversion_mag_x = None
#         new_vector.conversion_mag_y = None
    
#     # now we have delta_ed_es vectors, we do the final warp from ES-1 -> ES
#     # vector at ES frame will not have magnitudes, since we don't need to interpolate to warp to the next frame.
#     ES_minus_one_vect = all_new_vectors[-1]
#     tmp = Vector(ES_minus_one_vect.tail_x + ES_minus_one_vect.mag_x, ES_minus_one_vect.tail_y + ES_minus_one_vect.mag_y, 0, 0)
#     all_new_vectors.append(tmp)
        
#     return all_new_vectors

