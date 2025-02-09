from PIL import Image
import os
import numpy as np
import PIL
import time
import argparse
from pathlib import Path

'''
    Ensemble heat map from existing heatmaps.
    Run offline.
    
    Example:
    python utils/ensamble_heatmap.py /home/dlserver/Documents/data/GIA/masks 
    -m "1907Jul28_23-40-04,1908Aug04_14-57-46,1908Aug04_17-33-16,1908Aug04_17-34-29,1908Aug04_17-35-15,1908Aug04_17-36-30"
2
'''

filename_lst = ['TMI_CW', 'TMI_dX_CW', 'TMI_dY_CW', 'TMI_dZ_CW', 'RTP_CW', 'RTP_tiltder_CW', 'AS_CW']
N = 157288821

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', metavar='DIR', type=str,
                        default='/home/victoria/Documents/models/GIA/masks'
                        , help="Path to a folder with saved images and numpy binaries")
    parser.add_argument('-m', '--modelnames', type=str, default='', help="List of model names for voting")
    args = parser.parse_args()
    return args

def metrics(pred, y, case='', metrics_path=''):
    pred = pred.reshape(-1)
    y = y.reshape(-1)
    P = np.sum(y)
    N = len(y) - P
    TP = np.sum((pred == 1) & (y == 1))
    TN = np.sum((pred == 0) & (y == 0))
    FP = np.sum((pred == 1) & (y == 0))
    FN = np.sum((pred == 0) & (y == 1))
    precision, recall = -1, -1
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    line = '\n'
    line += case + '\n'
    line += 'N: {}, pos pred: {}, pos pred {:.2f}%, pos GT: {}, TP: {}, TN: {}, FP: {}, FN:{}'.\
                format(len(pred), np.sum(pred), (np.sum(pred) / len(pred)) * 100, P, TP, TN, FP, FN) + '\n'
    if P != 0 and N != 0:
        TPR = TP / P  # TPR = sensitivity
        TNR = TN / N  # TNR = recall = specificity
        FPR = FP / N
        FNR = FN / P
        acc = (TP + TN) / (P + N)
        F1 = (2 * TP) / (2 * TP + FP + FN)
        line += 'TPR: {:.2f}, TNR: {:.2f}, FPR: {:.2f}, FNR: {:.2f}, acc: {:.2f}, F1: {:.2f}, precision: {:.2f}, recall: {:.2f}'.\
            format(100 * TPR, 100 * TNR, 100 * FPR, 100 * FNR, 100 * acc, 100 * F1, 100 * precision, 100 * recall) + '\n'

    f = open(metrics_path, 'a+')
    f.write(line)
    print(line)

def ensamble_heatmap(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = Path(args.save_dir)
    voting_lst = ['maj', 'avg_prob', 'avg_thresh']
    modelname_str = args.modelnames
    modelname_lst = modelname_str.split(',')
    heatmap_lst = []
    metrics_path = Path(save_dir) / "metrics_{}.txt".format(modelname_str)

    for voting in voting_lst:
        for modelname in modelname_lst:
            heatmap_path = save_dir / 'heatmap_{}_{}.npz'.format(voting, modelname)
            if os.path.isfile(heatmap_path):
                heatmap = (np.load(heatmap_path)['arr_0']).astype(np.float16)
                heatmap_lst.append(heatmap)
            else:
                print('heatmap {} does not exist'.format(heatmap_path))
        heatmap_arr = np.stack(heatmap_lst).astype(np.float16)                          # (num_models, height, width)
        height, width = heatmap.shape[:2]

        # load pos mask
        pos_mask_path = save_dir / 'mask_pos_TMI_CW.npz'
        if os.path.isfile(pos_mask_path):
            pos_mask_bool = ((np.load(pos_mask_path)['arr_0']).astype(np.float16) > 0)*1
            pos_mask = pos_mask_bool * 150
        else:
            print('pos_mask {} does not exist'.format(pos_mask_path))

        # ensamble avg
        heatmap_ensamble_avg = np.mean(heatmap_arr, axis=0)
        im_heatmap = Image.fromarray(heatmap_ensamble_avg.astype(np.uint8))
        im_heatmap = im_heatmap.resize((width//10, height//10), PIL.Image.ANTIALIAS)
        im_heatmap.save(save_dir / 'im_heatmap_{}-avg_{}.png'.format(voting, modelname_str))
        np.savez(save_dir / 'heatmap_{}-avg_{}.npz'.format(voting, modelname_str), heatmap_ensamble_avg)

        heatmap_ensamble_avg_col = np.stack([heatmap_ensamble_avg, pos_mask, heatmap_ensamble_avg]).transpose(1,2,0)
        im_heatmap_col = Image.fromarray(heatmap_ensamble_avg_col.astype(np.uint8))
        im_heatmap_col = im_heatmap_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_col.save(save_dir / 'im_heatmap_{}-avg_col_{}.png'.format(voting, modelname_str))

        heatmap_ensamble_avg_bool = (heatmap_ensamble_avg > 100)*1
        metrics(heatmap_ensamble_avg_bool, pos_mask_bool,
                case='{}-avg_bool_col_{}_greater100.png'.format(voting, modelname_str), metrics_path=metrics_path)
        heatmap_ensamble_avg_bool *= 100
        heatmap_ensamble_avg_bool_col = np.stack([heatmap_ensamble_avg_bool, pos_mask, heatmap_ensamble_avg_bool]).transpose(1, 2, 0)
        im_heatmap_bool_col = Image.fromarray(heatmap_ensamble_avg_bool_col.astype(np.uint8))
        im_heatmap_bool_col = im_heatmap_bool_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_bool_col.save(save_dir / 'im_heatmap_{}-avg_bool_col_{}_greater100.png'.format(voting, modelname_str))

        heatmap_ensamble_avg_bool = (heatmap_ensamble_avg > 125) * 1
        metrics(heatmap_ensamble_avg_bool, pos_mask_bool,
                case='{}-avg_bool_col_{}_greater125.png'.format(voting, modelname_str), metrics_path=metrics_path)
        heatmap_ensamble_avg_bool *= 125
        heatmap_ensamble_avg_bool_col = np.stack([heatmap_ensamble_avg_bool, pos_mask, heatmap_ensamble_avg_bool]).transpose(1, 2, 0)
        im_heatmap_bool_col = Image.fromarray(heatmap_ensamble_avg_bool_col.astype(np.uint8))
        im_heatmap_bool_col = im_heatmap_bool_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_bool_col.save(save_dir / 'im_heatmap_{}-avg_bool_col_{}_greater125.png'.format(voting, modelname_str))

        heatmap_ensamble_avg_bool = (heatmap_ensamble_avg > 150) * 1
        metrics(heatmap_ensamble_avg_bool, pos_mask_bool,
                case='{}-avg_bool_col_{}_greater150.png'.format(voting, modelname_str), metrics_path=metrics_path)
        heatmap_ensamble_avg_bool *= 150
        heatmap_ensamble_avg_bool_col = np.stack([heatmap_ensamble_avg_bool, pos_mask, heatmap_ensamble_avg_bool]).transpose(1, 2, 0)
        im_heatmap_bool_col = Image.fromarray(heatmap_ensamble_avg_bool_col.astype(np.uint8))
        im_heatmap_bool_col = im_heatmap_bool_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_bool_col.save(save_dir / 'im_heatmap_{}-avg_bool_col_{}_greater150.png'.format(voting, modelname_str))

        # ensamble boolean all
        heatmap_ensamble_bool = (heatmap_arr > 0)*1
        heatmap_ensamble_all = np.ones_like(heatmap)
        for i in range(len(heatmap_ensamble_bool)):
            heatmap_ensamble_all *= heatmap_ensamble_bool[i]
        metrics(heatmap_ensamble_all, pos_mask_bool, case='{}-bool_all_{}.png'.format(voting, modelname_str),
                metrics_path=metrics_path)
        heatmap_ensamble_all *= 200
        im_heatmap = Image.fromarray(heatmap_ensamble_all.astype(np.uint8))
        im_heatmap = im_heatmap.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap.save(save_dir / 'im_heatmap_{}-bool_all_{}.png'.format(voting, modelname_str))
        np.savez(save_dir / 'heatmap_{}-bool_all_{}.npz'.format(voting, modelname_str), heatmap_ensamble_all)

        heatmap_ensamble_avg_col = np.stack([heatmap_ensamble_all, pos_mask, heatmap_ensamble_all]).transpose(1, 2, 0)
        im_heatmap_col = Image.fromarray(heatmap_ensamble_avg_col.astype(np.uint8))
        im_heatmap_col = im_heatmap_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_col.save(save_dir / 'im_heatmap_{}-bool_all_col_{}.png'.format(voting, modelname_str))

        # ensamble maj
        heatmap_ensamble_bool = (heatmap_arr > 0) * 1
        heatmap_ensamble_maj = np.sum(heatmap_ensamble_bool, axis=0)
        heatmap_ensamble_maj_bool = (heatmap_ensamble_maj > (len(heatmap_lst)//2))*1
        metrics(heatmap_ensamble_maj_bool, pos_mask_bool, case='{}-maj_{}.png'.format(voting, modelname_str),
                metrics_path=metrics_path)
        heatmap_ensamble_maj_bool *= 200
        im_heatmap = Image.fromarray(heatmap_ensamble_maj_bool.astype(np.uint8))
        im_heatmap = im_heatmap.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap.save(save_dir / 'im_heatmap_{}-maj_{}.png'.format(voting, modelname_str))
        np.savez(save_dir / 'heatmap_{}-maj_{}.npz'.format(voting, modelname_str), heatmap_ensamble_maj_bool)

        heatmap_ensamble_avg_col = np.stack([heatmap_ensamble_maj_bool, pos_mask, heatmap_ensamble_maj_bool]).transpose(1, 2, 0)
        im_heatmap_col = Image.fromarray(heatmap_ensamble_avg_col.astype(np.uint8))
        im_heatmap_col = im_heatmap_col.resize((width // 10, height // 10), PIL.Image.ANTIALIAS)
        im_heatmap_col.save(save_dir / 'im_heatmap_{}-maj_col_{}.png'.format(voting, modelname_str))

    print('heatmap with {} voting run time: {} min'.format(voting, (time.time() - t_i)/60))




if __name__ == '__main__':
    ensamble_heatmap()

    # from PIL import Image
    # p = 14
    # pol_idx_arr = np.arange(16)
    # # png_path = '/home/dlserver/Documents/data/GIA/experimental/im_mask_pos_TMI_CW.png'
    # # im = Image.open(png_path)
    # # img_arr = np.array(im)
    # file_path = '/home/dlserver/Documents/data/GIA/experimental/mask_pos_TMI_CW.npz'
    # img_arr = np.load(file_path)['arr_0']
    # img_arr_bool = ((img_arr >= pol_idx_arr[p]) & (img_arr < pol_idx_arr[p+1]))*200
    # im_heatmap_col = Image.fromarray(img_arr_bool.astype(np.uint8))
    # im_heatmap_col = im_heatmap_col.resize((1063, 1770), PIL.Image.ANTIALIAS)
    # im_heatmap_col.save('/home/dlserver/Documents/data/GIA/experimental/pol.png')































