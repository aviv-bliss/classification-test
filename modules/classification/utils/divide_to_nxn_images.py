from PIL import Image
import os
import numpy as np
import PIL
import time
import argparse
import pickle
from pathlib import Path

'''
    Load 7 magnetic field maps as np arrays, stack them and divide to positive 
    and negative nxn pixels images using a running window method. 
    n - size a small image, i - image index [0, N), p - GT polygon index [0, 14]. 
    First run positive case and then negative with max_neg twice number of pos samples.
    Run offline.
    
    Example:
    python utils/divide_to_nxn_images.py -d /home/dlserver/Documents/data/GIA/masks -n 50 -c pos --max_neg 500000 
'''

filename_lst = ['TMI_CW', 'TMI_dX_CW', 'TMI_dY_CW', 'TMI_dZ_CW', 'RTP_CW', 'RTP_tiltder_CW', 'AS_CW']
N = 157288821

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--save_dir', type=str, default='/home/victoria/Documents/models/GIA/masks',
                        help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-c', '--case', type=str, default='pos', help="Positive or negative GT")
    parser.add_argument('--max_neg', type=int, default=164530, help="Maximal number of negative GT samples")
    parser.add_argument('-n', type=int, default=10, help="Size of image nxn")
    parser.add_argument('-s', '--stride', type=int, default=1, help="Stride")
    args = parser.parse_args()
    return args


def load_np_array(save_dir, array_name='im_arr', num_items=7, save_img=False):
    '''
        Loads all 7 np arrays and returns them as a list
    '''
    assert num_items >= 1 and num_items <= len(filename_lst)
    array_lst = []
    for filename in filename_lst[:num_items]:
        file_path = os.path.join(save_dir, '{}_{}.npz'.format(array_name, filename))
        print('loading {}'.format(file_path))
        array = np.load(file_path)['arr_0']
        array_lst.append(array)

        # save images
        if save_img:
            img_file_path = os.path.join(save_dir, '{}_{}.png'.format(array_name, filename))
            img = Image.fromarray(array)
            img = img.resize((1063, 1770), PIL.Image.ANTIALIAS)
            img.save(img_file_path)
    print('loaded {} {} arrays of shape {}'.format(len(array_lst), array_name, array_lst[0].shape, array_lst[0]))

    return array_lst


def running_window(im_arr_stacked, mask, n, mask_dir, percent_true=100, stride=1, case='pos', max_neg=164530):
    '''
        Gets im_arr_stacked of shape (height, width, 7) and runs a window every
        'stride' pixels. Valid window is one, whose mask is True for 'percent_true'
        number of pixels in the window.
        Returns a list of all valid window arrays (N, n, n, 7).
        If 'case' is pos, returns in addition a list of length N of a polygon indices
        in the window mask as a np array.
    '''
    assert im_arr_stacked.shape[:2] == mask.shape
    valid_windows_lst, polygon_idxs_lst, idxs_lst = [], [], []
    height, width, ch = im_arr_stacked.shape
    samples_root_dir = Path(mask_dir).parent/'samples'
    if not os.path.exists(samples_root_dir):
        os.makedirs(samples_root_dir)
    samples_dir = samples_root_dir/'samples_n{}_{}'.format(n, case)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    count = 0
    num_valid_windows = 0
    mod = 1
    if case == 'neg':
        mod = N//max_neg
    print('N = {}, max_neg = {}, mod = {}'.format(N, max_neg, mod))

    for i in range(0, height-n, stride):
        for j in range(0, width-n, stride):
            window_mask = mask[i:i+n, j:j+n]
            window_mask_bool = window_mask > 0
            window_mask_num_pixels = window_mask.shape[0] * window_mask.shape[1]
            assert window_mask_num_pixels > 0
            window_mask_num_pos_pixels = np.sum(window_mask_bool)
            window_mask_percent_pos_pixels = (window_mask_num_pos_pixels/window_mask_num_pixels) * 100
            if window_mask_percent_pos_pixels >= percent_true:
                count += 1
                if count%mod==0:
                    window = im_arr_stacked[i:i+n, j:j+n]
                    valid_windows_lst.append(window)
                    window_mask_polygon_idxs = np.unique(window_mask)
                    polygon_idxs_lst.append(window_mask_polygon_idxs)
                    idxs_lst.append((i, j))
                    num_valid_windows += 1
                    print('\ncount = {}, num_valid_windows = {}: ({}, {})'.format(count, num_valid_windows, i, j))
                    # print('\n({}, {}): polygon_idxs: {}  \nmask: {}  \n\nwindow: {}'.format(i, j, window_mask_polygon_idxs,
                    #                                     window_mask, window.transpose(2, 0, 1)))

                    # save sample images
                    if num_valid_windows%100==0:
                        ch = [0]
                        path = os.path.join(samples_dir, 'ch0_i{}_j{}.png'.format(i, j))
                        if case == 'pos':
                            polyg = np.max(np.array(window_mask_polygon_idxs))
                            path = os.path.join(samples_dir, 'ch0_p{}_i{}_j{}.png'.format(polyg, i, j))
                        im_window = Image.fromarray(np.squeeze(window[:, :, ch[0]].astype(np.uint8)))
                        im_window = im_window.resize((100, 100), Image.NEAREST)
                        im_window.save(path)
                        ch = [0, 1, 3]
                        path = os.path.join(samples_dir, 'ch012_i{}_j{}.png'.format(i, j))
                        if case == 'pos':
                            polyg = np.max(np.array(window_mask_polygon_idxs))
                            path = os.path.join(samples_dir, 'ch012_p{}_i{}_j{}.png'.format(polyg, i, j))
                        im_window = Image.fromarray(window[:, :, ch])
                        im_window = im_window.resize((100, 100), Image.NEAREST)
                        im_window.save(path)
                        ch = [4, 5, 6]
                        path = os.path.join(samples_dir, 'ch456_i{}_j{}.png'.format(i, j))
                        if case == 'pos':
                            path = os.path.join(samples_dir, 'ch456_p{}_i{}_j{}.png'.
                                                format(window_mask_polygon_idxs, i, j))
                        im_window = Image.fromarray(window[:, :, ch])
                        im_window = im_window.resize((100, 100), Image.NEAREST)
                        im_window.save(path)
                        # with open(path, 'wb') as f:
                        #     pickle.dump(window, f)
                    if num_valid_windows > max_neg:
                        break

    print('found {} {}x{} valid_windows_lst for the {} case'.format(len(valid_windows_lst), n, n, case))

    if case == 'pos':
        return valid_windows_lst, idxs_lst, polygon_idxs_lst
    else:
        return valid_windows_lst, idxs_lst


def divide_to_nxn_images(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = args.save_dir
    case = args.case
    max_neg = args.max_neg
    stride = args.stride
    n = args.n
    print('n: {} case: {}'.format(n, case))

    # load images and stack them
    im_arr_lst = load_np_array(save_dir, array_name='im_arr', save_img=True)
    im_arr_stacked = np.stack(im_arr_lst)                                       # [7, height, width]
    im_arr_stacked = im_arr_stacked.transpose(1, 2, 0)                          # [height, width, 7]
    print('im_arr_stacked: {}'.format(im_arr_stacked.shape))

    # process pos/neg mask and pickle
    if case == 'pos':
        mask_pos = load_np_array(save_dir, array_name='mask_pos', num_items=1)[0]
        valid_windows_pos_lst, idxs_pos_lst, polygon_idxs_lst = running_window(im_arr_stacked, mask_pos, n, save_dir,
                                                                               percent_true=75, stride=stride)
        valid_windows_pos_lst_path = os.path.join(save_dir, 'valid_windows_pos_lst_{}.pkl'.format(n))
        polygon_idxs_lst_path = os.path.join(save_dir, 'polygon_idxs_lst_{}.pkl'.format(n))
        idxs_pos_lst_path = os.path.join(save_dir, 'idxs_pos_lst_{}.pkl'.format(n))
        with open(valid_windows_pos_lst_path, 'wb') as f:
            pickle.dump(valid_windows_pos_lst, f)
        with open(polygon_idxs_lst_path, 'wb') as f:
            pickle.dump(polygon_idxs_lst, f)
        with open(idxs_pos_lst_path, 'wb') as f:
            pickle.dump(idxs_pos_lst, f)
    elif case == 'neg':
        assert max_neg > 0
        mask_neg = load_np_array(save_dir, array_name='mask_neg', num_items=1)[0]
        valid_windows_neg_lst, idxs_neg_lst = running_window(im_arr_stacked, mask_neg, n, save_dir, percent_true=100,
                                                             stride=stride, case='neg', max_neg=max_neg)
        valid_windows_neg_lst_path = os.path.join(save_dir, 'valid_windows_neg_lst_{}.pkl'.format(n))
        idxs_neg_lst_path = os.path.join(save_dir, 'idxs_neg_lst_{}.pkl'.format(n))
        with open(valid_windows_neg_lst_path, 'wb') as f:
            pickle.dump(valid_windows_neg_lst, f)
        with open(idxs_neg_lst_path, 'wb') as f:
            pickle.dump(idxs_neg_lst, f)
    else:
        print('wrong case arg: {}'.format(case))

    print('run time: {} min'.format((time.time() - t_i)/60))






if __name__ == '__main__':
    divide_to_nxn_images()

    # # open pickled array
    # filename = 'valid_windows_pos_lst_20'
    # idx = 55
    # path = '/home/dlserver/Documents/models/GIA/masks/{}.pkl'.format(filename)
    # with open(path, 'rb') as f:
    #     valid_windows_lst = pickle.load(f)
    # window = valid_windows_lst[idx]
    # ch = [0, 2, 3]
    # im_window = Image.fromarray(window[:, :, ch])
    # im_window = im_window.resize((100, 100), Image.NEAREST)
    # print('window[:, :, {}]: {} \n{}'.format(ch, window.shape, window[:, :, ch].transpose(2, 0, 1)))
    # im_window.save('/home/dlserver/Documents/models/GIA/masks/{}_{}.png'.format(filename, idx))
































