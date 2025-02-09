import os
import numpy as np
import time
import argparse
from pathlib import Path
from math import *
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
import dataloaders.img_transforms as transforms
from QuantDisc.utils.make_subsets import load_pickled_lists
from QuantDisc.utils.make_heatmap import load_model

import torch
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

'''
    Sample 100 windows from each polygon and 10000 negative windows.
    Run inference on them and take the activations after global pooling (feture vector (FV) of size 32).
    Run similarity metric between a FV from one polygon and all other polygons and negative FVs.
    Order according to similarity and get top-N for each positive sample.
    Run offline.
    (for running from terminal:
       $ export PYTHONPATH=/home/dlserver/Documents/projects/POCs/QuantDisc:$PYTHONPATH   )

    Example:
    python utils/feature_vectors_similarity.py /home/dlserver/Documents/models/QuantDisc/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar 
    -d /home/dlserver/Documents/data/QuantDisc/masks -n 50 -f 480
    
    python utils/feature_vectors_similarity.py /home/victoria/Dropbox/Neural_Networks/Trained_models/QuantDisc/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar 
    -d /media/victoria/d/data/QuantDisc/masks -n 50 -f 480
    
'''

num_polygons = 14
max_num_pos_windows = 100    # per polygon
max_num_neg_windows = 10000
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
    parser.add_argument('-d', '--mask_dir', type=str, default='/home/victoria/Documents/models/QuantDisc/masks',
                        help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-n', type=int, default=50, help="Sample's size n")
    parser.add_argument('-f', '--feat_vec_num', type=int, default=4, help="Best feature vector's number")
    args = parser.parse_args()
    return args


def sample_max_num_of_pos_neg_windows(mask_dir, n, max_num_pos_windows, max_num_neg_windows):
    all_pos_windows_path = mask_dir / 'all_pos_windows_arr_{}.npz'.format(n)
    all_neg_windows_path = mask_dir / 'all_neg_windows_arr_{}.npz'.format(n)
    all_pos_windows_idx_arr_path = mask_dir / 'all_pos_windows_idx_arr_{}.npz'.format(n)
    all_neg_windows_idx_arr_path = mask_dir / 'all_neg_windows_idx_arr_{}.npz'.format(n)

    # load max_num_pos_windows per polygon windows
    if not os.path.isfile(all_pos_windows_path):
        # load pos and neg lists
        valid_windows_pos_lst, idxs_pos_lst, polygon_idxs_lst = load_pickled_lists(mask_dir, n, case='pos')
        print('pos samples: {}'.format(len(valid_windows_pos_lst)))

        # make dict of windows for every polygon
        num_windows_in_polygon = np.zeros(num_polygons + 1).astype(np.int64)
        all_polyg_idx_arr = np.ones((num_polygons, len(valid_windows_pos_lst) // 2)).astype(np.int64) * (-1)
        for idx, polygon_idxs in enumerate(polygon_idxs_lst):
            for i in range(len(polygon_idxs)):
                if polygon_idxs[i] > 0:
                    all_polyg_idx_arr[polygon_idxs[i] - 1, num_windows_in_polygon[polygon_idxs[i]]] = idx
                    num_windows_in_polygon[polygon_idxs[i]] += 1
        num_windows_in_polygon = num_windows_in_polygon[1:]
        print('number of windows for every polygon: {}'.format(str(num_windows_in_polygon)))

        # sample max_num_pos_windows for each polygon
        max_idx = 0
        all_pos_windows_arr = np.zeros((num_polygons, max_num_pos_windows, n, n, valid_windows_pos_lst[0].shape[2])).astype(np.float32)
        all_pos_windows_idx_arr = np.zeros((num_polygons, max_num_pos_windows, 2)).astype(np.int64)
        for i in range(num_polygons):
            p_polygon_idxs = all_polyg_idx_arr[i]
            first_minus1_idx = np.where(all_polyg_idx_arr[i] == -1)[0][0]
            if first_minus1_idx > max_idx:
                max_idx = first_minus1_idx
            if first_minus1_idx > 100:
                p_polygon_idxs = p_polygon_idxs[:first_minus1_idx]
# !!!
                # np.random.shuffle(p_polygon_idxs)                      ## !!! canceled shuffle !!!
                p_polygon_idxs = p_polygon_idxs[:max_num_pos_windows]
                for w, idx in enumerate(p_polygon_idxs):
                    all_pos_windows_arr[i, w] = valid_windows_pos_lst[idx]
                    all_pos_windows_idx_arr[i, w] = np.array(idxs_pos_lst[idx])
        print('max_idx: {}'.format(max_idx))
        all_pos_windows_arr = all_pos_windows_arr[:, 0:max_idx, :, :, :]
        all_pos_windows_idx_arr = all_pos_windows_idx_arr[:, 0:max_idx, :]
        np.savez(all_pos_windows_path, all_pos_windows_arr)
        np.savez(all_pos_windows_idx_arr_path, all_pos_windows_idx_arr)
    else:
        all_pos_windows_arr = np.load(all_pos_windows_path)['arr_0']
        all_pos_windows_idx_arr = np.load(all_pos_windows_idx_arr_path)['arr_0']
        print('loaded {}: {}'.format(all_pos_windows_path, all_pos_windows_arr.shape))

    # load max_num_neg_windows negative windows
    if not os.path.isfile(all_neg_windows_path):
        valid_windows_neg_lst, idxs_neg_lst = load_pickled_lists(mask_dir, n, case='neg')
        print('neg samples: {}'.format(len(valid_windows_neg_lst)))
        all_neg_windows_arr = np.zeros((max_num_neg_windows, n, n, valid_windows_pos_lst[0].shape[2])).astype(np.float32)
        all_neg_windows_idx_arr = np.zeros((max_num_neg_windows, 2)).astype(np.int64)
        neg_windows_idxs = np.arange(len(valid_windows_neg_lst))
        np.random.shuffle(neg_windows_idxs)
        neg_windows_idxs = neg_windows_idxs[:max_num_neg_windows]
        for w, idx in enumerate(neg_windows_idxs):
            all_neg_windows_arr[w] = valid_windows_neg_lst[idx]
            all_neg_windows_idx_arr[w] = idxs_neg_lst[idx]
        np.savez(all_neg_windows_path, all_neg_windows_arr)
        np.savez(all_neg_windows_idx_arr_path, all_neg_windows_idx_arr)
    else:
        all_neg_windows_arr = np.load(all_neg_windows_path)['arr_0']
        all_neg_windows_idx_arr = np.load(all_neg_windows_idx_arr_path)['arr_0']
        print('loaded {}: {}'.format(all_neg_windows_path, all_neg_windows_arr.shape))

    return all_pos_windows_arr, all_neg_windows_arr, all_pos_windows_idx_arr, all_neg_windows_idx_arr


def get_feature_vectors(model, all_pos_windows_arr, all_neg_windows_arr, mask_dir, modelname, batch_size=200):
    totensor = transforms.Compose([transforms.ToTensor(), ])
    feature_vecs_pos_path = mask_dir / 'feature_vecs_pos_{}.npz'.format(modelname)
    feature_vecs_neg_path = mask_dir / 'feature_vecs_neg_{}.npz'.format(modelname)

    if not os.path.isfile(feature_vecs_pos_path):
        # run an inference on pos windows and take the layer after global pooling
        window_batch, feature_vecs_pos_lst = [], []
        batch_idx = 0
        for pm1 in range(num_polygons):
            for i, window in enumerate(all_pos_windows_arr[pm1]):
                if np.sum(window) > 0:
                    window_batch.append(window)
                    batch_idx += 1

                # get prediction from the model for every batch
                if batch_idx == max_num_pos_windows:
                    window_batch_np = np.stack(window_batch)
                    data_t = totensor(window_batch_np)  # (batch_size, num_channels, height, width)
                    data_t = Variable(data_t)
                    if use_cuda:
                        data_t = data_t.cuda()
                    output = model(data_t)
                    feature_vecs = np.array(output.data.cpu().numpy()[:])
                    feature_vecs_pos_lst.append(feature_vecs)
                    batch_idx = 0
                    del window_batch
                    window_batch = []
        feature_vecs_pos_arr = np.vstack(feature_vecs_pos_lst)
        np.savez(feature_vecs_pos_path, feature_vecs_pos_arr)
        print('feature_vecs_pos_arr saved as {}'.format(feature_vecs_pos_path))
    else:
        feature_vecs_pos_arr = np.load(feature_vecs_pos_path)['arr_0']
        print('loaded {}: {}'.format(feature_vecs_pos_path, feature_vecs_pos_arr.shape))
    num_features = feature_vecs_pos_arr.shape[-1]
    feature_vecs_pos_arr = feature_vecs_pos_arr.reshape(-1, max_num_pos_windows, num_features)

    if not os.path.isfile(feature_vecs_neg_path):
        # run an inference on neg windows and take the layer after global pooling
        window_batch, feature_vecs_neg_lst = [], []
        batch_idx = 0
        for i, window in enumerate(all_neg_windows_arr):
            if np.sum(window) > 0:
                window_batch.append(window)
                batch_idx += 1

            # get prediction from the model for every batch
            if batch_idx == batch_size:
                window_batch_np = np.stack(window_batch)
                data_t = totensor(window_batch_np)  # (batch_size, num_channels, height, width)
                data_t = Variable(data_t)
                if use_cuda:
                    data_t = data_t.cuda()
                output = model(data_t)
                feature_vecs = np.array(output.data.cpu().numpy()[:])
                feature_vecs_neg_lst.append(feature_vecs)
                batch_idx = 0
                del window_batch
                window_batch = []
        feature_vecs_neg_arr = np.vstack(feature_vecs_neg_lst)
        np.savez(feature_vecs_neg_path, feature_vecs_neg_arr)
        print('feature_vecs_neg_arr saved as {}'.format(feature_vecs_neg_path))
    else:
        feature_vecs_neg_arr = np.load(feature_vecs_neg_path)['arr_0']
        print('loaded {}: {}'.format(feature_vecs_neg_path, feature_vecs_neg_arr.shape))

    return feature_vecs_pos_arr, feature_vecs_neg_arr


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def compute_similarity(feature_vecs_pos_arr, feature_vecs_neg_arr, all_pos_windows_idx_arr, all_neg_windows_idx_arr,
                       mask_dir, modelname):

    simil_matrix_path = mask_dir / 'simil_matrix_{}.npz'.format(modelname)
    simil_matrix_idxs_path = mask_dir / 'simil_matrix_idxs_{}.npz'.format(modelname)
    num_pos_feature_vecs = feature_vecs_pos_arr.shape[0] * feature_vecs_pos_arr.shape[1]
    num_neg_feature_vecs = feature_vecs_neg_arr.shape[0]
    simil_matrix = np.ones((num_pos_feature_vecs, num_pos_feature_vecs + num_neg_feature_vecs)).astype(np.float32) * (-1)
    simil_matrix_idxs = np.ones((num_pos_feature_vecs, num_pos_feature_vecs + num_neg_feature_vecs, 4)).astype(np.int32) * (-1)

    # similarity between pos feature vectors
    if not os.path.isfile(simil_matrix_path):
        # pos-pos
        num_polygons = len(feature_vecs_pos_arr)
        for p_i in range(num_polygons):
            polyg_arr = np.arange(num_polygons)
#!!! changed !!!
            # polyg_arr = np.delete(polyg_arr, p_i)         ## !!! commented !!!
            for v_i, feature_vec_i in enumerate(feature_vecs_pos_arr[p_i]):
                for p_j in polyg_arr:
                    for v_j, feature_vec_j in enumerate(feature_vecs_pos_arr[p_j]):
                        i = p_i * max_num_pos_windows + v_i
                        j = p_j * max_num_pos_windows + v_j
                        simil_matrix[i, j] = cosine_similarity(feature_vec_i, feature_vec_j)
                        simil_matrix_idxs[i, j] = np.array([all_pos_windows_idx_arr[p_i, v_i, 0],
                                                           all_pos_windows_idx_arr[p_i, v_i, 1],
                                                           all_pos_windows_idx_arr[p_j, v_j, 0],
                                                           all_pos_windows_idx_arr[p_j, v_j, 1]])   # (x_i, y_i, x_j, y_j)

        # pos-neg
        for p_i in range(num_polygons):
            for v_i, feature_vec_i in enumerate(feature_vecs_pos_arr[p_i]):
                for v_j, feature_vec_j in enumerate(feature_vecs_neg_arr):
                    i = p_i * max_num_pos_windows + v_i
                    j = num_pos_feature_vecs + v_j
                    simil_matrix[i, j] = cosine_similarity(feature_vec_i, feature_vec_j)
                    simil_matrix_idxs[i, j] = np.array([all_pos_windows_idx_arr[p_i, v_i, 0],
                                                       all_pos_windows_idx_arr[p_i, v_i, 1],
                                                       all_neg_windows_idx_arr[v_j, 0],
                                                       all_neg_windows_idx_arr[v_j, 1]])      # (x_i, y_i, x_j, y_j)

        np.savez(simil_matrix_path, simil_matrix)
        np.savez(simil_matrix_idxs_path, simil_matrix_idxs)
        print('simil_matrix saved as {}: {}'.format(simil_matrix_path, simil_matrix.shape))
    else:
        simil_matrix = np.load(simil_matrix_path)['arr_0']
        simil_matrix_idxs = np.load(simil_matrix_idxs_path)['arr_0']
        print('simil_matrix loaded: {}'.format(simil_matrix.shape))

    return simil_matrix, simil_matrix_idxs


def similarity_rating(simil_matrix, feature_vecs_pos_arr, mask_dir, modelname):
    top5_path = mask_dir / 'top5_{}.txt'.format(modelname)
    top10_path = mask_dir / 'top10_{}.txt'.format(modelname)
    top20_path = mask_dir / 'top20_{}.txt'.format(modelname)
    top100_path = mask_dir / 'top100_{}.txt'.format(modelname)
    if os.path.isfile(top5_path): os.remove(top5_path)
    if os.path.isfile(top10_path): os.remove(top10_path)
    if os.path.isfile(top20_path): os.remove(top20_path)
    if os.path.isfile(top100_path): os.remove(top100_path)

    # if not (os.path.isfile(top5_path) and os.path.isfile(top10_path) and os.path.isfile(top100_path)):
    simil_matrix_sorted = np.zeros_like(simil_matrix)
    simil_matrix_sorted_idxs = np.zeros_like(simil_matrix)
    num_pos_feature_vecs = feature_vecs_pos_arr.shape[0] * feature_vecs_pos_arr.shape[1]
    idx_lst = list(np.arange(simil_matrix.shape[1]))
    idx_str_lst = []
    for idx in idx_lst:
        if idx < num_pos_feature_vecs:
            idx_str_lst.append('*{}*'.format(idx))
        else:
            idx_str_lst.append('{}'.format(idx))
    top5, top10, top20, top100 = [], [], [], []
    f5 = open(top5_path, 'a+')
    f10 = open(top10_path, 'a+')
    f20 = open(top20_path, 'a+')
    f100 = open(top100_path, 'a+')
    for i in range(len(simil_matrix)):
        simil_matrix_i = simil_matrix[i]
        simil_matrix_i_sorted_idxs = np.argsort(simil_matrix_i)[::-1]
        simil_matrix_sorted_idxs[i] = simil_matrix_i_sorted_idxs
        simil_matrix_sorted[i] = simil_matrix_i[simil_matrix_i_sorted_idxs]
        # top5_i = sorted([idx_str_lst[idx] for idx in simil_matrix_i_sorted_idxs[:5] if idx < num_pos_feature_vecs])
        top5_i = ([idx_str_lst[idx] for idx in simil_matrix_i_sorted_idxs[:5]])
        top5_i_val = ([simil_matrix_i[idx] for idx in simil_matrix_i_sorted_idxs[:5]])
        top5.append(top5_i)
        if len(top5_i) > 0:
            f5.write('vector {} is similar to vectors: {}\n'.format(i, str(top5_i)))
            f5.write('                                 {}\n'.format(str(top5_i_val)))
        # top10_i = sorted([idx for idx in simil_matrix_i_sorted_idxs[:10] if idx < num_pos_feature_vecs])
        top10_i = ([idx_str_lst[idx] for idx in simil_matrix_i_sorted_idxs[:10]])
        top10_i_val = ([simil_matrix_i[idx] for idx in simil_matrix_i_sorted_idxs[:10]])
        top10.append(top10_i)
        if len(top10_i) > 0:
            f10.write('vector {} is similar to vectors: {}\n'.format(i, str(top10_i)))
            f10.write('                                 {}\n'.format(str(top10_i_val)))
        # top20_i = sorted([idx for idx in simil_matrix_i_sorted_idxs[:20] if idx < num_pos_feature_vecs])
        top20_i = ([idx_str_lst[idx] for idx in simil_matrix_i_sorted_idxs[:20]])
        top20_i_val = ([simil_matrix_i[idx] for idx in simil_matrix_i_sorted_idxs[:20]])
        top20.append(top20_i)
        if len(top20_i) > 0:
            f20.write('vector {} is similar to vectors: {}\n'.format(i, str(top20_i)))
            f20.write('                                 {}\n'.format(str(top20_i_val)))
        # top100_i = sorted([idx for idx in simil_matrix_i_sorted_idxs[:100] if idx < num_pos_feature_vecs])
        top100_i = ([idx_str_lst[idx] for idx in simil_matrix_i_sorted_idxs[:100]])
        top100_i_val = ([simil_matrix_i[idx] for idx in simil_matrix_i_sorted_idxs[:100]])
        top100.append(top100_i)
        if len(top100_i) > 0:
            f100.write('vector {} is similar to vectors: {}\n'.format(i, str(top100_i)))
            f100.write('                                 {}\n'.format(str(top100_i_val)))
    print('saved top-5,10 and 100 to {}'.format(mask_dir))

    return simil_matrix_sorted_idxs


def feature_vectors_similarity(args=None):
    t_i = time.time()
    only_feature_vecs = False
    if args is None:
        args = getArgs()
        model_path = Path(args.model_path)
        mask_dir = Path(args.mask_dir)
        n = args.n
        feat_vec_num = args.feat_vec_num
    else:
        model_path, mask_dir, n = args
        model_path = Path(model_path)
        mask_dir = Path(mask_dir)
        only_feature_vecs = True
    modelname = model_path.parent.parent.basename()
    print('model_path: {}\nmask_dir: {}\nn: {}'.format(model_path, mask_dir, n))

    # load max_num of pos and neg windows
    all_pos_windows_arr, all_neg_windows_arr, all_pos_windows_idx_arr, all_neg_windows_idx_arr = \
             sample_max_num_of_pos_neg_windows(mask_dir, n, max_num_pos_windows, max_num_neg_windows)
    all_pos_windows_idx_arr = all_pos_windows_idx_arr[~np.all(all_pos_windows_idx_arr.reshape(len(all_pos_windows_idx_arr),-1) == 0, axis=1)]

    # load model
    model, _ = load_model(model_path)

    # get feature vectors
    feature_vecs_pos_arr, feature_vecs_neg_arr = get_feature_vectors(model, all_pos_windows_arr,
                                                      all_neg_windows_arr, mask_dir, modelname)
    if only_feature_vecs:
        return feature_vecs_pos_arr.reshape(-1, feature_vecs_pos_arr.shape[-1])

    # compute similarity
    simil_matrix, simil_matrix_idxs = compute_similarity(feature_vecs_pos_arr, feature_vecs_neg_arr,
                                          all_pos_windows_idx_arr, all_neg_windows_idx_arr, mask_dir, modelname)

    # rate pairs of FVs according to their similarity
    simil_matrix_sorted_idxs = similarity_rating(simil_matrix, feature_vecs_pos_arr, mask_dir, modelname)

    # load pos mask
    pos_mask_path = mask_dir / 'mask_pos_TMI_CW.npz'
    if os.path.isfile(pos_mask_path):
        pos_mask_bool = ((np.load(pos_mask_path)['arr_0']).astype(np.float16) > 0) * 1
        pos_mask = pos_mask_bool * 100
    else:
        print('pos_mask {} does not exist'.format(pos_mask_path))

    # represent best feature vectors' similarities on a map
    feat_vec_map = np.zeros_like(pos_mask).astype(np.float32)
    sim_thresh = 0.01
    num_windows = 0
    for j, similarity in enumerate(simil_matrix[feat_vec_num]):
        _, _, x, y = simil_matrix_idxs[feat_vec_num, j]
        if x > 0 and y > 0 and similarity > sim_thresh:
            num_windows += 1
            for k in range(n):
                for l in range(n):
                    feat_vec_map[x+k, y+l] = similarity - sim_thresh
    print('feat_vec_map: feat_vec_num: {}, num_windows: {}'.format(feat_vec_num, num_windows))
    feat_vec_map = (feat_vec_map*255).astype(np.uint8)
    feat_vec_map_col = np.stack([feat_vec_map, pos_mask, feat_vec_map]).transpose(1, 2, 0)
    feat_vec_map_col = Image.fromarray(feat_vec_map_col.astype(np.uint8))
    # feat_vec_map_col = feat_vec_map_col.resize((1063*10, 1770*10), PIL.Image.ANTIALIAS)
    feat_vec_map_col.save(mask_dir / 'im_feat_vec_map_{}_{}.png'.format(modelname, feat_vec_num))

    # represent top-10 feature vectors' similarities on a map
    num_windows = 0
    sim_coords_top10_path = mask_dir / 'sim_coords_{}_{}_top10.txt'.format(modelname, feat_vec_num)
    f = open(sim_coords_top10_path, 'a+')
    f.write('sample    pixel_coord i      pixel_coord j         map_coord x          map_coord y\n')
    feat_vec_map_top10 = np.zeros_like(pos_mask).astype(np.float32)
    for j in simil_matrix_sorted_idxs[feat_vec_num, :10]:
        similarity = simil_matrix[int(feat_vec_num), int(j)]
        _, _, x, y = simil_matrix_idxs[int(feat_vec_num), int(j)]
        if x > 0 and y > 0:
            num_windows += 1
            x_map = x_min + x * pixel_size
            y_map = y_min + y * pixel_size
            f.write('{}       {}               {}              {:.1f}            {:.1f}\n'.
                    format(int(j), y + n // 2, x + n // 2, x_map, y_map))
            for k in range(n):
                for l in range(n):
                    feat_vec_map_top10[x + k, y + l] = similarity
    feat_vec_map_top10 = (feat_vec_map_top10 * 255).astype(np.uint8)
    feat_vec_map_col = np.stack([feat_vec_map_top10, pos_mask, feat_vec_map_top10]).transpose(1, 2, 0)
    feat_vec_map_col = Image.fromarray(feat_vec_map_col.astype(np.uint8))
    feat_vec_map_col = feat_vec_map_col.resize((1063*2, 1770*2), Image.ANTIALIAS)
    feat_vec_map_col.save(mask_dir / 'im_feat_vec_map_top10_{}_{}_small.png'.format(modelname, feat_vec_num))
    print('feat_vec_map_top10: feat_vec_num: {}, num_windows: {}'.format(feat_vec_num, num_windows))

    # represent top-20 feature vectors similarities on a map
    num_windows = 0
    sim_coords_top20_path = mask_dir / 'sim_coords_{}_{}_top20.txt'.format(modelname, feat_vec_num)
    f = open(sim_coords_top20_path, 'a+')
    f.write('sample    pixel_coord i      pixel_coord j         map_coord x          map_coord y\n')
    feat_vec_map_top20 = np.zeros_like(pos_mask).astype(np.float32)
    for j in simil_matrix_sorted_idxs[feat_vec_num, :20]:
        similarity = simil_matrix[int(feat_vec_num), int(j)]
        _, _, x, y = simil_matrix_idxs[int(feat_vec_num), int(j)]
        if x > 0 and y > 0:
            num_windows += 1
            x_map = x_min + x * pixel_size
            y_map = y_min + y * pixel_size
            f.write('{}       {}               {}              {:.1f}            {:.1f}\n'.
                    format(int(j), y + n // 2, x + n // 2, x_map, y_map))
            for k in range(n):
                for l in range(n):
                    feat_vec_map_top20[x + k, y + l] = similarity
    feat_vec_map_top20 = (feat_vec_map_top20 * 255).astype(np.uint8)
    feat_vec_map_col = np.stack([feat_vec_map_top20, pos_mask, feat_vec_map_top20]).transpose(1, 2, 0)
    feat_vec_map_col = Image.fromarray(feat_vec_map_col.astype(np.uint8))
    feat_vec_map_col = feat_vec_map_col.resize((1063*2, 1770*2), Image.ANTIALIAS)
    feat_vec_map_col.save(mask_dir / 'im_feat_vec_map_top20_{}_{}_small.png'.format(modelname, feat_vec_num))
    print('feat_vec_map_top20: feat_vec_num: {}, num_windows: {}'.format(feat_vec_num, num_windows))

    # present f.v. 480 and its neighbor feature vectors on a map
    ## (!sample's shuffle inside each polygon Should be cancelled cancelled and similarity calculation with its own polygon allowed!)
    ## pass (no need to do now)

    print('feature_vecs run time: {} min'.format((time.time() - t_i)/60))





if __name__ == '__main__':
    feature_vectors_similarity()
































