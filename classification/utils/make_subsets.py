from PIL import Image
import os
import numpy as np
import time
import argparse
import pickle

'''
    Load a list of positive and negative nxn images and polygon indices for positive GT. 
    Make subsets of polygons: 9/3/3 for train/test/valid modes and make lists of positive
    train/test/valid images.
    Make lists of negative images subsets in approximate proportion 70%/15%/15%.
    Pickle pos and neg images together for every mode.
    Run offline.

    Example:
    python utils/make_subsets.py -d /home/dlserver/Documents/data/GIA -n 50
    or  
    python utils/make_subsets.py -d /home/dlserver/Documents/data/GIA --n_lst "30,40,50,60,100" 
'''
# cross-valid 1
train_polygs_lst = [1, 2, 3, 5, 7, 9, 8, 10, 11, 12]
test_polygs_lst = [4, 13]
val_polygs_lst = [6, 14]

# # cross-valid 2
# train_polygs_lst = [5, 6, 7, 9, 12, 13]
# test_polygs_lst = [10, 14]
# val_polygs_lst = [4, 11]

# # cross-valid 3
# train_polygs_lst = [4, 5, 7, 9, 10, 14]
# test_polygs_lst = [11, 13]
# val_polygs_lst = [6, 12]

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root_dir', type=str, default='/home/dlserver/Documents/data/GIA',
                        help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-n', type=int, default=50, help="Sample's size n")
    parser.add_argument('--n_lst', type=str, default='', help="List of sample sizes for making train/test/val sets")
    args = parser.parse_args()
    return args


def load_pickled_lists(masks_dir, n, case='pos', save_img=False):
    '''
        Loads a pickled list
    '''
    valid_windows_lst_path = os.path.join(masks_dir, 'valid_windows_{}_lst_{}.pkl'.format(case, n))
    with open(valid_windows_lst_path, 'rb') as f:
        valid_windows_lst = pickle.load(f)
        print('loaded {}: {}'.format(valid_windows_lst_path, len(valid_windows_lst)))
    if save_img:
        window = valid_windows_lst[0]
        ch = [0, 2, 3]
        im_window = Image.fromarray(window[:, :, ch])
        im_window = im_window.resize((100, 100), Image.NEAREST)
        print('window[:, :, {}]: {} \n{}'.format(ch, window.shape, window[:, :, ch].transpose(2, 0, 1)))
        im_window.save('/home/dlserver/Documents/models/GIA/masks/valid_windows_{}_{}_0.png'.format(case, n))

    idxs_lst_path = os.path.join(masks_dir, 'idxs_{}_lst_{}.pkl'.format(case, n))
    with open(idxs_lst_path, 'rb') as f:
        idxs_lst = pickle.load(f)
        print('loaded {}: {}'.format(idxs_lst_path, len(idxs_lst)))

    if case == 'pos':
        polygon_idxs_lst_path = os.path.join(masks_dir, 'polygon_idxs_lst_{}.pkl'.format(n))
        with open(polygon_idxs_lst_path, 'rb') as f:
            polygon_idxs_lst = pickle.load(f)
            print('loaded {}: {}'.format(polygon_idxs_lst_path, len(polygon_idxs_lst)))
        return valid_windows_lst, idxs_lst, polygon_idxs_lst
    else:
        return valid_windows_lst, idxs_lst


def make_subsets(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    root_dir = args.root_dir
    masks_dir = os.path.join(root_dir, 'masks')
    save_dir = os.path.join(root_dir, 'sets')
    n = args.n
    n_lst = [n]
    suffix = str(n) #+ '_3'
    if args.n_lst != '':
        n_lst = list(map(int, args.n_lst.split(',')))
        suffix = args.n_lst
    assert n_lst[0] != 0
    print('n_lst: {}'.format(n_lst))
    train_img_pos_all_path = os.path.join(save_dir, 'train_img_pos_{}.pkl'.format(suffix))
    test_img_pos_all_path = os.path.join(save_dir, 'test_img_pos_{}.pkl'.format(suffix))
    val_img_pos_all_path = os.path.join(save_dir, 'val_img_pos_{}.pkl'.format(suffix))
    train_img_neg_all_path = os.path.join(save_dir, 'train_img_neg_{}.pkl'.format(suffix))
    test_img_neg_all_path = os.path.join(save_dir, 'test_img_neg_{}.pkl'.format(suffix))
    val_img_neg_all_path = os.path.join(save_dir, 'val_img_neg_{}.pkl'.format(suffix))
    train_img_all_lst_pos, test_img_all_lst_pos, val_img_all_lst_pos = [], [], []
    train_img_all_lst_neg, test_img_all_lst_neg, val_img_all_lst_neg = [], [], []

    for nn in n_lst:
        print('nn: {}'.format(nn))
        suff = '{}'.format(nn)  #'{}_3'.format(nn)

        # load pos train/test/val sets if exist or make ones
        train_img_pos_path = os.path.join(save_dir, 'train_img_pos_{}.pkl'.format(suff))
        test_img_pos_path = os.path.join(save_dir, 'test_img_pos_{}.pkl'.format(suff))
        val_img_pos_path = os.path.join(save_dir, 'val_img_pos_{}.pkl'.format(suff))
        if os.path.isfile(train_img_pos_path) and os.path.isfile(test_img_pos_path) and os.path.isfile(val_img_pos_path):
            with open(train_img_pos_path, 'rb') as f:
                train_img_lst_pos = pickle.load(f)
                print('loaded {}'.format(train_img_pos_path), len(train_img_lst_pos))
            with open(test_img_pos_path, 'rb') as f:
                test_img_lst_pos = pickle.load(f)
            with open(val_img_pos_path, 'rb') as f:
                val_img_lst_pos = pickle.load(f)
        else:
            # load pos lists
            valid_windows_pos_lst, idxs_pos_lst, polygon_idxs_lst = load_pickled_lists(masks_dir, nn, case='pos',
                                                                                       save_img=False)

            # make lists of positive train/test/valid images
            train_img_lst_pos, test_img_lst_pos, val_img_lst_pos = [], [], []
            num_imgs_lst = np.zeros(15)
            for idx, polygon_idxs in enumerate(polygon_idxs_lst):
                for i in range(len(polygon_idxs)):
                    num_imgs_lst[polygon_idxs[i]] += 1
                    if polygon_idxs[i] in train_polygs_lst:
                        train_img_lst_pos.append(valid_windows_pos_lst[idx])
                    elif polygon_idxs[i] in test_polygs_lst:
                        test_img_lst_pos.append(valid_windows_pos_lst[idx])
                    elif polygon_idxs[i] in val_polygs_lst:
                        val_img_lst_pos.append(valid_windows_pos_lst[idx])
        train_img_all_lst_pos += train_img_lst_pos
        test_img_all_lst_pos += test_img_lst_pos
        val_img_all_lst_pos += val_img_lst_pos
        print('number of windows for every polygon: {}'.format(str(num_imgs_lst)))

        # load neg train/test/val sets if exist or make ones
        train_img_neg_path = os.path.join(save_dir, 'train_img_neg_{}.pkl'.format(suff))
        test_img_neg_path = os.path.join(save_dir, 'test_img_neg_{}.pkl'.format(suff))
        val_img_neg_path = os.path.join(save_dir, 'val_img_neg_{}.pkl'.format(suff))
        if os.path.isfile(train_img_neg_path) and os.path.isfile(test_img_neg_path) and os.path.isfile(val_img_neg_path):
            with open(train_img_neg_path, 'rb') as f:
                train_img_lst_neg = pickle.load(f)
                print('loaded {}'.format(train_img_neg_path), len(train_img_lst_neg))
            with open(test_img_neg_path, 'rb') as f:
                test_img_lst_neg = pickle.load(f)
            with open(val_img_neg_path, 'rb') as f:
                val_img_lst_neg = pickle.load(f)
        else:
            # load neg lists
            valid_windows_neg_lst, idxs_neg_lst = load_pickled_lists(masks_dir, nn, case='neg', save_img=False)

            # make lists of shuffled negative images subsets in proportion 70%/15%/15%
            N_neg = len(valid_windows_neg_lst)
            idx_arr = np.arange(N_neg)
            np.random.shuffle(idx_arr)
            num_imgs_train, num_imgs_test, num_imgs_val = int(N_neg*0.7), int(N_neg*0.15), int(N_neg*0.15)
            train_img_lst_neg, test_img_lst_neg, val_img_lst_neg = [], [], []
            for i, idx in enumerate(idx_arr):
                if i < num_imgs_train:
                    train_img_lst_neg.append(valid_windows_neg_lst[idx])
                elif i >= num_imgs_train and i < num_imgs_train+num_imgs_test:
                    test_img_lst_neg.append(valid_windows_neg_lst[idx])
                elif i >= num_imgs_train+num_imgs_test:
                    val_img_lst_neg.append(valid_windows_neg_lst[idx])
        train_img_all_lst_neg += train_img_lst_neg
        test_img_all_lst_neg += test_img_lst_neg
        val_img_all_lst_neg += val_img_lst_neg

        print('nn {}: train_img_lst_pos: {}, test_img_lst_pos: {}, val_img_lst_pos: {}, total: {}'.
              format(nn, len(train_img_lst_pos), len(test_img_lst_pos), len(val_img_lst_pos),
                     len(train_img_lst_pos) + len(test_img_lst_pos) + len(val_img_lst_pos)))
        print('nn {}: train_img_lst_neg: {}, test_img_lst_neg: {}, val_img_lst_neg: {}, total: {}'.
              format(nn, len(train_img_lst_neg), len(test_img_lst_neg), len(val_img_lst_neg),
                     len(train_img_lst_neg) + len(test_img_lst_neg) + len(val_img_lst_neg)))

    print('train_img_all_lst_pos: {}, test_img_all_lst_pos: {}, val_img_all_lst_pos: {}, total: {}'.
          format(len(train_img_all_lst_pos), len(test_img_all_lst_pos), len(val_img_all_lst_pos),
                 len(train_img_all_lst_pos) + len(test_img_all_lst_pos) + len(val_img_all_lst_pos)))
    print('train_img_all_lst_neg: {}, test_img_all_lst_neg: {}, val_img_all_lst_neg: {}, total: {}'.
          format(len(train_img_all_lst_neg), len(test_img_all_lst_neg), len(val_img_all_lst_neg),
                 len(train_img_all_lst_neg) + len(test_img_all_lst_neg) + len(val_img_all_lst_neg)))

    # pickle pos and neg images together for every mode.
    with open(train_img_pos_all_path, 'wb') as f:
        pickle.dump(train_img_all_lst_pos, f)
        print('{} saved'.format(train_img_pos_all_path))
    with open(test_img_pos_all_path, 'wb') as f:
        pickle.dump(test_img_all_lst_pos, f)
    with open(val_img_pos_all_path, 'wb') as f:
        pickle.dump(val_img_all_lst_pos, f)
    with open(train_img_neg_all_path, 'wb') as f:
        pickle.dump(train_img_all_lst_neg, f)
    with open(test_img_neg_all_path, 'wb') as f:
        pickle.dump(test_img_all_lst_neg, f)
    with open(val_img_neg_all_path, 'wb') as f:
        pickle.dump(val_img_all_lst_neg, f)
    print('run time: {} min'.format((time.time() - t_i)/60))






if __name__ == '__main__':
    make_subsets()


























