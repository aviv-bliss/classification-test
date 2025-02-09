from PIL import Image
import os
import numpy as np
import PIL
from skimage import measure
import time


def keep_largest_connected_components(mask, idx_thresh):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    idx_thresh defines an index in sorted area (small to large) to keep keep items from.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    binary_img = (mask == 1)
    blobs = measure.label(binary_img, connectivity=1)
    props = measure.regionprops(blobs)
    area = np.array([ele.area for ele in props])
    area_sorted_idxs = np.argsort(area)
    area_sorted = area[area_sorted_idxs]
    area_sorted_idxs_thresh = area_sorted_idxs[idx_thresh:]
    label_arr = np.array([props[idx].label for idx in area_sorted_idxs_thresh])
    for label in label_arr:
        out_img[blobs == label] = 1

    return out_img


def remove_small_cc(idx_thresh):
    '''
        Remolve small connected components from an image
    '''
    t_i = time.time()
    filename = 'heatmap_maj-avg_1908Aug04_14-57-46'

    # load pos mask
    pos_mask_path = '/home/dlserver/Documents/data/GIA/masks/mask_pos_TMI_CW.npz'
    if os.path.isfile(pos_mask_path):
        pos_mask_bool = ((np.load(pos_mask_path)['arr_0']).astype(np.float16) > 0) * 1
        pos_mask = pos_mask_bool * 150
    else:
        print('pos_mask {} does not exist'.format(pos_mask_path))

    # save_dir = Path('/home/dlserver/Documents/data/GIA/results/good_val')
    # filename = 'im_heatmap_maj-avg_bool_col_1907Jul28_23-40-04,1908Aug04_14-57-46_greater100.png'
    # im = Image.open(save_dir/filename)
    # img_arr = np.array(im)
    heatmap_path = '/home/dlserver/Documents/data/GIA/results/heatmap_maj-avg_1908Aug04_14-57-46.npz'
    if os.path.isfile(heatmap_path):
        heatmap = ((np.load(heatmap_path)['arr_0']).astype(np.float16) > 150) * 200
        heatmap_col = np.stack([heatmap, pos_mask, heatmap]).transpose(1, 2, 0)
        im_heatmap_col = Image.fromarray(heatmap_col.astype(np.uint8))
        im_heatmap_col = im_heatmap_col.resize((1063, 1770), PIL.Image.ANTIALIAS)
        im_heatmap_col.save('/home/dlserver/Documents/data/GIA/results/{}.png'.format(filename))
    else:
        print('pos_mask {} does not exist'.format(heatmap_path))



    # mask = (img_arr[:, :, 0] > 0) * 1
    mask = (heatmap > 0) * 1
    out_img = keep_largest_connected_components(mask, idx_thresh=idx_thresh)
    np.savez('/home/dlserver/Documents/data/GIA/results/{}_cleaned_{}.npz'.format(filename, idx_thresh), out_img)
    out_img *= 200
    out_img_col = np.stack([out_img, pos_mask, out_img]).transpose(1,2,0)
    im_heatmap_col = Image.fromarray(out_img_col.astype(np.uint8))
    im_heatmap_col = im_heatmap_col.resize((1063, 1770), PIL.Image.ANTIALIAS)
    im_heatmap_col.save('/home/dlserver/Documents/data/GIA/results/{}_cleaned_{}.png'.format(filename, idx_thresh))

    print('run time: {} min'.format((time.time() - t_i)/60))



if __name__ == '__main__':
    remove_small_cc(idx_thresh=1050)






















