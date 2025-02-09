from PIL import Image
import os
import numpy as np
import PIL
import time
import argparse

'''
    Load tiff (magnetic field maps) and shape files (copper deposits GT polygons) and 
    create image np arrays and GT positive/negative masks of presence/absence of copper deposits.
    Run offline.
    Example:
    python utils/make_valid_GT_mask.py --save_dir /home/dlserver/Documents/data/GIA/masks --filename TMI_CW
'''

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_dir', type=str, default='/home/victoria/Documents/models/GIA/masks',
                        help="Path to a folder to save images and numpy binaries")
    parser.add_argument('-t', '--tif_dir', type=str, default='/home/victoria/Documents/data/GIA/benchmark/4010_data_set/4010_data_set/data_4010_igrf/DATA_4010_IGRF',
                        help="Path to a folder of tiff images")
    parser.add_argument('-p', '--shapefile_path', type=str,
                        default='/home/victoria/Documents/data/GIA/benchmark/4010_data_set/4010_data_set/Delineations/20190618_4010_depos.shp',
                        help="Path to a shape file")
    parser.add_argument('-f', '--filename', type=str, default='TMI_CW', help="Tiff filename")
    args = parser.parse_args()
    return args


def make_GT_masks(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = args.save_dir
    filename = args.filename
    print('\n\n ############################# {} #################################'.format(filename))

    ####################### read a tif image ######################################
    im_arr = np.load(os.path.join(save_dir, 'im_arr_RTP_tiltder_CW.npz'))['arr_0']

    print('tif file: {}'.format(im_arr.shape))
    blank_points = np.nonzero(im_arr==255)
    blank_points_np = np.array(blank_points).transpose(1, 0)
    print('blank points: {}, {}'.format(len(blank_points[0]), len(blank_points[1])))
    mask_neg_with_blank_path = os.path.join(save_dir, 'mask_neg_with_blank_{}.npz'.format(filename))
    if not os.path.isfile(mask_neg_with_blank_path):
        mask_neg_with_blank = np.ones_like(im_arr)
        for i in range(len(blank_points_np)):
            idx_i = blank_points_np[i][0]
            idx_j = blank_points_np[i][1]
            mask_neg_with_blank[idx_i, idx_j] = 0
        np.savez(mask_neg_with_blank_path, mask_neg_with_blank)
        print('saved mask_neg_with_blank: {}'.format(mask_neg_with_blank.shape))
    else:
        mask_neg_with_blank = np.load(mask_neg_with_blank_path)['arr_0']
        print('loaded mask_neg_with_blank: {}'.format(mask_neg_with_blank.shape))
    nonblank_points = np.nonzero(im_arr<255)
    print('non-blank points: {}, {}'.format(len(nonblank_points[0]), len(nonblank_points[1])))

    np.savez(os.path.join(save_dir, 'mask_valid_{}.npz'.format(filename)), mask_neg_with_blank)
    im_neg = Image.fromarray(mask_neg_with_blank*255)
    im_neg = im_neg.resize((1063, 1770), PIL.Image.ANTIALIAS)
    im_neg.save(os.path.join(save_dir, 'mask_valid_{}.png'.format(filename)))

    print('run time: {} min'.format((time.time() - t_i)/60))




if __name__ == '__main__':
    make_GT_masks()























