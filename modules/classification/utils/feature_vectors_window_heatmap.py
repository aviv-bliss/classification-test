from PIL import Image
import os
import numpy as np
import PIL
import time
import argparse
import pickle
from pathlib import Path
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
from utils.make_heatmap import load_model
from utils.divide_to_nxn_images import load_np_array
import dataloaders.img_transforms as transforms
from utils.feature_vectors_similarity import cosine_similarity, feature_vectors_similarity

import torch
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

'''
    Load 7 magnetic field maps as np arrays and stack them; 
    load negative mask (without blanks).
    Run a window nxn on the valid area (in which mask is positive) with stride 's'. 
    Create list of predicted values for every valid pixel.
    Do some voting ('average' or 'minimum one positive') and make a heatmap.
    Show the probability map.
    Run offline.
    (for running from terminal:
       $ export PYTHONPATH=/home/dlserver/Documents/projects/POCs/QuantDisc:$PYTHONPATH   )
    
    python utils/feature_vectors_window_heatmap.py /home/victoria/Dropbox/Neural_Networks/Trained_models/QuantDisc/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar 
    -d /media/victoria/d/data/QuantDisc/masks -n 50 -s 50 -f 52
'''

x_min = 380088.3
y_min = 7344018.3
pixel_size = 12.7
seed = 117
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='DIR', help="Path to a model ckpt for inference")
    parser.add_argument('-d', '--save_dir', type=str, default='/home/victoria/Documents/models/QuantDisc/masks',
                        help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-s', '--stride', type=int, default=1, help="Stride")
    parser.add_argument('-n', type=int, default=50, help="Sample's size n")
    parser.add_argument('-f', '--feat_vec_num', type=int, help="Best feature vector's number")
    args = parser.parse_args()
    return args


def running_window(img_arr, mask, model_path, feature_vec_input, sim_arr_path, percent_true=100, stride=1):
    '''
        Gets img_arr of shape (height, width, 7) and runs a window every
        'stride' pixels. Valid window is one, whose mask is True for 'percent_true'
        number of pixels in the window.
        Adds the predicted similarity to the given feature vector to all pixels in the window
        (appends it to a list in all_pix_dict).
        Returns all_pix_dict.
    '''
    assert img_arr.shape[:2] == mask.shape
    model, FLAGS = load_model(model_path)
    totensor = transforms.Compose([transforms.ToTensor(), ])
    height, width, ch = img_arr.shape
    window_batch, ij_lst = [], []
    num_windows, num_valid_windows, batch_idx, num_sim_greater095 = 0, 0, 0, -1
    feat_vec_map = np.zeros_like(mask).astype(np.float32)
    n = FLAGS.height
    sim_lst = []
    sim_arr_path.basename()
    sim_coords_path = sim_arr_path.parent / 'sim_coords.txt'
    f = open(sim_coords_path, 'a+')
    f.write('i    pixel_coord i      pixel_coord j         map_coord x          map_coord y\n')
    arr = np.arange(31)
    np.random.shuffle(arr)
    # arr = arr[:20]
    nn = 0

    for i in range(0, height-n, stride):
        for j in range(0, width-n, stride):
            num_windows += 1
            if num_windows%10000==0:
                print('processing window {}, num_valid_windows: {}'.format(num_windows, num_valid_windows))
            window_mask = mask[i:i+n, j:j+n]
            window_mask_bool = window_mask > 0
            window_mask_num_pixels = window_mask.shape[0] * window_mask.shape[1]
            window_mask = None
            assert window_mask_num_pixels > 0
            window_mask_num_pos_pixels = np.sum(window_mask_bool)
            window_mask_percent_pos_pixels = (window_mask_num_pos_pixels/window_mask_num_pixels) * 100
            window_mask_num_pos_pixels = None
            if window_mask_percent_pos_pixels >= percent_true:
                num_valid_windows += 1
                window = img_arr[i:i + n, j:j + n]
                window = window.reshape(FLAGS.height, FLAGS.width, FLAGS.num_channels)
                window_batch.append(window)
                window = None
                ij_lst.append((i, j))
                batch_idx += 1

                # get prediction from the model for every batch
                if batch_idx == FLAGS.batch_size:
                    window_batch_np = np.stack(window_batch)
                    data_t = totensor(window_batch_np)         # (batch_size, num_channels, height, width)
                    window_batch_np = None
                    data_t = Variable(data_t)
                    if use_cuda:
                        data_t = data_t.cuda()
                    output = model(data_t)
                    feature_vec_out_lst = output.data.cpu().numpy()[:]
                    similarity_lst = [cosine_similarity(feature_vec_input, feature_vec_out) for feature_vec_out in
                                      feature_vec_out_lst]

                    assert len(ij_lst) == len(similarity_lst)
                    for idx, (ii, jj) in enumerate(ij_lst):
                        similarity = similarity_lst[idx]
                        if similarity == 1: #> 0.95:
                            num_sim_greater095 += 1
                            if num_sim_greater095 in arr:
                                sim_lst.append(np.array([similarity, ii, jj]))
                                nn += 1
                                for k in range(n):
                                    for l in range(n):
                                        feat_vec_map[ii + k, jj + l] = similarity
                                        if (k == n//2) and (l == n//2):
                                            x = x_min + jj*pixel_size
                                            y = y_min + ii*pixel_size
                                            f.write('{}       {}               {}              {:.1f}            {:.1f}\n'.
                                                    format(nn, ii+n//2, jj+n//2, x, y))
                                print('{}: ({},{}) sim={}'.format(num_sim_greater095, ii, jj, similarity))

                    batch_idx = 0
                    del window_batch
                    del ij_lst
                    window_batch, ij_lst = [], []

    sim_arr = np.array(sim_lst)
    np.savez(sim_arr_path, sim_arr)
    print('found {} samples with similarity = 1.0 and saved as {}'.format(sim_arr.shape, sim_arr_path))

    return sim_arr, feat_vec_map


def feature_vectors_window_heatmap(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = Path(args.save_dir)
    model_path = Path(args.model_path)
    modelname = model_path.parent.parent.basename()
    stride = args.stride
    n = args.n
    feat_vec_num = args.feat_vec_num
    sim_arr_path = save_dir / 'sim_arr_{}_{}'.format(modelname, feat_vec_num)
    mask_valid = load_np_array(save_dir, array_name='mask_valid', num_items=1)[0]
    height, width = mask_valid.shape[:2]
    print('feat_vec_num: {}, stride: {}, model: {}'.format(feat_vec_num, stride, model_path))

    # load best feature vector
    args = [model_path, save_dir, n]
    feature_vecs_pos_arr = feature_vectors_similarity(args)
    feature_vec_input = feature_vecs_pos_arr[feat_vec_num]

    if not os.path.isdir(sim_arr_path):
        # load images and stack them
        im_arr_lst = load_np_array(save_dir, array_name='im_arr', save_img=True)
        im_arr_stacked = np.stack(im_arr_lst)                           # [7, height, width]
        im_arr_stacked = im_arr_stacked.transpose(1, 2, 0)                 # [height, width, 7]
        print('im_arr_stacked: {}'.format(im_arr_stacked.shape))

        # inference on all running windows in a map --> list of predicted values for every valid pixel
        sim_arr, feat_vec_map = running_window(im_arr_stacked, mask_valid, model_path, feature_vec_input, sim_arr_path,
                                               stride=stride)
        print('running_window run time: {} min'.format((time.time() - t_i) / 60))
    else:
        sim_arr = np.load(sim_arr_path)['arr_0']
        print('loaded sim_arr {}: {}'.format(sim_arr_path, sim_arr.shape))

    # load pos mask
    pos_mask_path = save_dir / 'mask_pos_TMI_CW.npz'
    if os.path.isfile(pos_mask_path):
        pos_mask_bool = ((np.load(pos_mask_path)['arr_0']).astype(np.float16) > 0) * 1
        pos_mask = pos_mask_bool * 100
    else:
        print('pos_mask {} does not exist'.format(pos_mask_path))

    feat_vec_map = (feat_vec_map * 200).astype(np.uint8)
    feat_vec_map_col = np.stack([feat_vec_map, pos_mask, feat_vec_map]).transpose(1, 2, 0)
    feat_vec_map_col = Image.fromarray(feat_vec_map_col.astype(np.uint8))
    # feat_vec_map_col = feat_vec_map_col.resize((1063*10, 1770*10), PIL.Image.ANTIALIAS)
    feat_vec_map_col.save(save_dir / 'im_feat_vec_map_window_{}_{}.png'.format(modelname, feat_vec_num))



if __name__ == '__main__':
    feature_vectors_window_heatmap()





























