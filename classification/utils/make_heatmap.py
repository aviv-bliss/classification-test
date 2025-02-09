from PIL import Image
import os
import numpy as np
import PIL
import time
import argparse
import pickle
from pathlib import Path
from collections import OrderedDict
import json

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
from QuantDisc.utils.divide_to_nxn_images import load_np_array
from QuantDisc.utils.auxiliary import load_model_and_weights, softmax
import QuantDisc.dataloaders.img_transforms as transforms

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
       $ export PYTHONPATH=/home/dlserver/Documents/projects/POCs/GIA:$PYTHONPATH   )

    Example:
    python utils/make_heatmap.py /home/dlserver/Documents/models/GIA/1908Aug04_14-57-46/ckpts/model_ckpt_best.pth.tar 
    -d /home/dlserver/Documents/data/GIA/masks -p 0.68749 -s 10 
'''

filename_lst = ['TMI_CW', 'TMI_dX_CW', 'TMI_dY_CW', 'TMI_dZ_CW', 'RTP_CW', 'RTP_tiltder_CW', 'AS_CW']
seed = 117
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar='DIR', help="Path to a model ckpt for inference")
    parser.add_argument('-d', '--save_dir', type=str, default='/home/victoria/Documents/models/GIA/masks',
                        help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-p', '--prob_thresh', type=float, default=0.5, help="Probability threshold")
    parser.add_argument('-s', '--stride', type=int, default=1, help="Stride")
    args = parser.parse_args()
    return args


def load_model(model_path, prob_thresh=0.5):
    train_dir = model_path.parent.parent
    config_path = train_dir / 'args_test.json'
    assert os.path.isfile(config_path)
    with open(config_path, 'rt') as r:
        config = json.load(r)

    class FLAGS():
        def __init__(self):
            self.load_ckpt = model_path
            self.batch_size = config['batch_size']
            self.train_dir = train_dir
            self.height = config['height']
            self.width = config['width']
            self.data_loader = config['data_loader']
            self.model = config['model']
            self.metric = config['metric']
            self.num_channels = config['num_channels']
            self.num_classes = config['num_classes']
            self.n_filters = config['n_filters']
            self.batch_norm = config['batch_norm']
            self.version = config['version']
            self.kernel_size = config['kernel_size']
            self.stride = config['stride']
            self.fc = config['fc']
            self.balanced = config['balanced']
            self.seed = config['seed']
            self.prob_thresh = prob_thresh
    FLAGS = FLAGS()

    # load model
    models_loaded, models, model_names, n_iter, n_epoch = load_model_and_weights(model_path, FLAGS, use_cuda,
                                        model=config['model'], ckpts_dir=train_dir / 'ckpts', train=False)
    model = models[0]
    model.eval()
    return model, FLAGS


def save_imgs(window, num_valid_windows, save_dir, num_windows, probability, i, j, n, ch):
    # save windows as images
    print('num_windows: {}, num_valid_windows: {}'.format(num_windows, num_valid_windows))
    samples_dir = save_dir.parent / 'samples/windows_n{}'.format(n)
    if not os.path.exists(samples_dir):
        print("Creating folder {}".format(samples_dir))
        os.makedirs(samples_dir)
    if ch == 1 or ch == 3:
        im_window = Image.fromarray(np.squeeze(window.astype(np.uint8)))
        im_window.save(samples_dir/'i{}_j{}_p{:.3f}.png'.format(i, j, probability[-1]))
    else:
        np.savez(samples_dir/'i{}_j{}_p{:.3f}.npz'.format(i, j, probability[-1]), window)
    print('a sample ({}, {}) saved to {}'.format(i, j, samples_dir))


def running_window(img_arr, mask, model_path, prob_thresh, save_dir, modelname, percent_true=100, stride=1,
                   save_imgs=False):
    '''
        Gets img_arr of shape (height, width, 7) and runs a window every
        'stride' pixels. Valid window is one, whose mask is True for 'percent_true'
        number of pixels in the window.
        Adds the predicted probability to all pixels in the window (appends it to a list
        in all_pix_dict).
        Returns all_pix_dict.
    '''
    assert img_arr.shape[:2] == mask.shape
    # img_arr = img_arr[:, :, :config['num_channels']]
    model, FLAGS = load_model(model_path, prob_thresh)
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
                print('processing window {}, num_valid_windows: {}, num_dicts {}'.format(num_windows, num_valid_windows,
                                                                                         num_dicts))
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

                    logits = output.data.cpu().numpy()
                    probability = np.array([softmax(logits[i]) for i in range(len(logits))])[:, 1]

                    # add the predicted probabilty to every pixel in the window
                    idx = 0
                    assert len(ij_lst) == len(probability)
                    for (ii, jj) in ij_lst:
                        for p in range(ii, ii+n):
                            for q in range(jj, jj+n):
                                key = '{}_{}'.format(p, q)
                                if not key in all_pix_dict:
                                    all_pix_dict[key] = [probability[idx]]
                                    num_pixels += 1

                                    if num_pixels % 1000000 == 0:
                                        all_pix_dict_dir = save_dir / 'all_pix_dict_{}'.format(modelname)
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
                                    all_pix_dict[key].append(probability[idx])
                        idx += 1
                    batch_idx = 0
                    del window_batch
                    del ij_lst
                    window_batch, ij_lst = [], []

                if save_imgs and num_valid_windows % 1000000 == 0:
                    save_imgs(window, num_valid_windows, save_dir, num_windows, probability, i, j, n, FLAGS.num_channels)

    print('found {} {}x{} valid windows; number of all_pix_dict parts: {}'.format(num_valid_windows, FLAGS.height,
                                                                                  FLAGS.height, num_dicts))


def make_heatmap(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = Path(args.save_dir)
    model_path = Path(args.model_path)
    modelname = model_path.parent.parent.basename()
    prob_thresh = args.prob_thresh
    stride = args.stride
    all_pix_dict_dir = save_dir / 'all_pix_dict_{}'.format(modelname)
    mask_valid = load_np_array(save_dir, array_name='mask_valid', num_items=1)[0]
    height, width = mask_valid.shape[:2]
    print('stride: {}, model: {}'.format(stride, model_path))

    if not os.path.isdir(all_pix_dict_dir):
        # load images and stack them
        im_arr_lst = load_np_array(save_dir, array_name='im_arr', save_img=True)
        im_arr_stacked = np.stack(im_arr_lst)                                       # [7, height, width]
        im_arr_stacked = im_arr_stacked.transpose(1, 2, 0)                          # [height, width, 7]
        print('im_arr_stacked: {}'.format(im_arr_stacked.shape))

        # inference on all running windows in a map --> list of predicted values for every valid pixel
        running_window(im_arr_stacked, mask_valid, model_path, prob_thresh, save_dir, modelname, stride=stride)
        print('running_window run time: {} min'.format((time.time() - t_i)/60))
        t_i = time.time()

    # loop over all dict parts
    num_dicts = len(os.listdir(all_pix_dict_dir))
    heatmap_avg_prob = np.zeros(mask_valid.shape[:2])
    heatmap_avg_thresh = np.zeros(mask_valid.shape[:2])
    heatmap_maj = np.zeros(mask_valid.shape[:2])
    for dict_idx in range(num_dicts):
        all_pix_dict_path = all_pix_dict_dir / 'all_pix_dict_{}.pkl'.format(dict_idx)
        with open(all_pix_dict_path, 'rb') as f:
            all_pix_dict = pickle.load(f)
            print('{} loaded'.format(all_pix_dict_path))

        # voting for every pixel
        for key, i_j_prob_lst in all_pix_dict.items():
            i, j = list(map(int, key.split('_')))
            # voting 'avg_prob':
            heatmap_avg_prob[i, j] = np.mean(np.array(i_j_prob_lst)) * 255
            # voting 'avg_thresh':
            i_j_pred_thresh = (np.array(i_j_prob_lst) > prob_thresh) * 1
            heatmap_avg_thresh[i, j] = np.mean(i_j_pred_thresh) * 255
            # voting 'maj':
            i_j_pred_thresh = (np.array(i_j_prob_lst) > prob_thresh) * 1
            heatmap_maj[i, j] = (np.sum(i_j_pred_thresh) >= (len(i_j_prob_lst)/2)) * 255
        all_pix_dict.clear()

    # save the heatmaps
    im_prob_map_avg = Image.fromarray(heatmap_avg_prob.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width//10, height//10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_heatmap_avg_prob_{}.png'.format(modelname))
    np.savez(save_dir / 'heatmap_avg_prob_{}.npz'.format(modelname), heatmap_avg_prob)

    im_prob_map_avg = Image.fromarray(heatmap_avg_thresh.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_heatmap_avg_thresh_{}.png'.format(modelname))
    np.savez(save_dir / 'heatmap_avg_thresh_{}.npz'.format(modelname), heatmap_avg_thresh)

    im_prob_map_avg = Image.fromarray(heatmap_maj.astype(np.uint8))
    im_prob_map_avg = im_prob_map_avg.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    im_prob_map_avg.save(save_dir / 'im_heatmap_maj_{}.png'.format(modelname))
    np.savez(save_dir / 'heatmap_maj_{}.npz'.format(modelname), heatmap_maj)

    # heatmap_avg_prob_thresh = (np.array(heatmap_avg_prob) > prob_thresh) * 1
    # im_avg_prob_thresh = Image.fromarray(heatmap_avg_prob_thresh.astype(np.uint8))
    # im_avg_prob_thresh = im_avg_prob_thresh.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
    # im_avg_prob_thresh.save(save_dir / 'im_heatmap_avg_prob_thresh_{}.png'.format(modelname))
    # np.savez(save_dir / 'heatmap_avg_prob_thresh_{}.npz'.format(modelname), heatmap_avg_prob_thresh)

    print('heatmap with run time: {} min'.format((time.time() - t_i)/60))




if __name__ == '__main__':
    make_heatmap()
































