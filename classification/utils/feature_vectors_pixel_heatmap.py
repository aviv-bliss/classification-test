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
    
    python utils/feature_vectors_pixel_heatmap.py /home/dlserver/Documents/models/QuantDisc/1907Jul28_23-40-04/ckpts/model_ckpt_best.pth.tar -d /home/dlserver/Documents/data/QuantDisc/masks -n 30 -s 7 -f 611
    
    python utils/feature_vectors_pixel_heatmap.py /home/dlserver/Documents/models/QuantDisc/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar -d /home/dlserver/Documents/data/QuantDisc/masks -n 50 -s 15 -f 458
    
    python utils/feature_vectors_pixel_heatmap.py /home/victoria/Dropbox/Neural_Networks/Trained_models/QuantDisc/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar 
    -d /media/victoria/d/data/QuantDisc/masks -n 50 -s 15 -f 4
'''

num_polygons = 14
max_num_pos_windows = 100      # per polygon
max_num_neg_windows = 10000
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


def running_window(img_arr, mask, model_path, feature_vec_input, all_pix_dict_dir, percent_true=100, stride=1):
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
    num_windows, num_valid_windows, batch_idx, num_pixels, num_dicts = 0, 0, 0, 0, 0
    all_pix_dict = OrderedDict()
    n = FLAGS.height

    for i in range(0, height-n, stride):
        for j in range(0, width-n, stride):
    # for i in range(7482, 7600, stride):
    #     for j in range(6653, 6800, stride):
            num_windows += 1
            if num_windows%10000==0:
                print('processing window {}, num_valid_windows: {}, num_dicts {}'.format(num_windows,
                                                                num_valid_windows, num_dicts))
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

                    # add the predicted probabilty to every pixel in the window
                    idx = 0
                    assert len(ij_lst) == len(similarity_lst)
                    for (ii, jj) in ij_lst:
                        for p in range(ii, ii+n):
                            for q in range(jj, jj+n):
                                key = '{}_{}'.format(p, q)
                                if not key in all_pix_dict:
                                    all_pix_dict[key] = [similarity_lst[idx]]
                                    num_pixels += 1

                                    if num_pixels % 1000000 == 0:
                                        if not os.path.exists(all_pix_dict_dir):
                                            print("Creating folder {}".format(all_pix_dict_dir))
                                            os.makedirs(all_pix_dict_dir)
                                        all_pix_dict_path = all_pix_dict_dir / 'all_pix_dict_{}.pkl'.format(num_dicts)
                                        with open(all_pix_dict_path, 'wb') as f:
                                            pickle.dump(all_pix_dict, f)
                                            print('all_pix_dict_{} is saved'.format(num_dicts))
                                        all_pix_dict.clear()
                                        all_pix_dict = OrderedDict()
                                        num_dicts += 1
                                else:
                                    all_pix_dict[key].append(similarity_lst[idx])
                        idx += 1
                    batch_idx = 0
                    del window_batch
                    del ij_lst
                    window_batch, ij_lst = [], []

    print('found {} {}x{} valid windows; number of all_pix_dict parts: {}'.format(num_valid_windows, FLAGS.height,
                                                                                  FLAGS.height, num_dicts))

def feature_vectors_pixel_heatmap(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = Path(args.save_dir)
    model_path = Path(args.model_path)
    modelname = model_path.parent.parent.basename()
    stride = args.stride
    n = args.n
    feat_vec_num = args.feat_vec_num
    all_pix_dict_dir = save_dir / 'all_pix_dict_feature_vec_{}_{}'.format(modelname, feat_vec_num)
    mask_valid = load_np_array(save_dir, array_name='mask_valid', num_items=1)[0]
    height, width = mask_valid.shape[:2]
    print('feat_vec_num: {}, stride: {}, model: {}'.format(feat_vec_num, stride, model_path))

    # load best feature vector
    args = [model_path, save_dir, n]
    feature_vecs_pos_arr = feature_vectors_similarity(args)
    feature_vec_input = feature_vecs_pos_arr[feat_vec_num]

    if not os.path.isdir(all_pix_dict_dir):
        # load images and stack them
        im_arr_lst = load_np_array(save_dir, array_name='im_arr', save_img=True)
        im_arr_stacked = np.stack(im_arr_lst)                           # [7, height, width]
        im_arr_stacked = im_arr_stacked.transpose(1, 2, 0)                 # [height, width, 7]
        print('im_arr_stacked: {}'.format(im_arr_stacked.shape))

        # inference on all running windows in a map --> list of predicted values for every valid pixel
        running_window(im_arr_stacked, mask_valid, model_path, feature_vec_input, all_pix_dict_dir, stride=stride)
        print('running_window run time: {} min'.format((time.time() - t_i) / 60))
        t_i = time.time()

    # loop over all dict parts
    num_dicts = len(os.listdir(all_pix_dict_dir))
    heatmap_avg = np.zeros(mask_valid.shape[:2])
    heatmap_min = np.zeros(mask_valid.shape[:2])
    heatmap_max = np.zeros(mask_valid.shape[:2])
    for dict_idx in range(num_dicts):
        all_pix_dict_path = all_pix_dict_dir / 'all_pix_dict_{}.pkl'.format(dict_idx)
        with open(all_pix_dict_path, 'rb') as f:
            all_pix_dict = pickle.load(f)
            print('{} loaded'.format(all_pix_dict_path))

        # voting for every pixel
        for key, i_j_prob_lst in all_pix_dict.items():
            i, j = list(map(int, key.split('_')))
            # avg_prob
            heatmap_avg[i, j] = np.mean(np.array(i_j_prob_lst)) * 255
            # min
            heatmap_min[i, j] = np.min(np.array(i_j_prob_lst)) * 255
            # max
            heatmap_max[i, j] = np.max(np.array(i_j_prob_lst)) * 255
        all_pix_dict.clear()

    # save the heatmaps
    im_prob_map_avg = Image.fromarray(heatmap_avg.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_feature_vec_heatmap_avg_{}_{}.png'.format(modelname, feat_vec_num))
    np.savez(save_dir / 'feature_vec_heatmap_avg_{}_{}.npz'.format(modelname, feat_vec_num), heatmap_avg)

    im_prob_map_avg = Image.fromarray(heatmap_min.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_feature_vec_heatmap_min_{}_{}.png'.format(modelname, feat_vec_num))
    np.savez(save_dir / 'feature_vec_heatmap_min_{}_{}.npz'.format(modelname, feat_vec_num), heatmap_min)

    im_prob_map_avg = Image.fromarray(heatmap_max.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_feature_vec_heatmap_max_{}_{}.png'.format(modelname, feat_vec_num))
    np.savez(save_dir / 'feature_vec_heatmap_max_{}_{}.npz'.format(modelname, feat_vec_num), heatmap_max)

    print('feature_vec {} heatmap run time: {} min'.format(feat_vec_num, (time.time() - t_i) / 60))





if __name__ == '__main__':
    feature_vectors_pixel_heatmap()





























