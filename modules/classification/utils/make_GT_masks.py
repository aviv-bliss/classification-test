from PIL import Image
import gdal
import ogr
import os
import rasterio as rio
import numpy as np
import geopandas as gpd
import matplotlib.path as mpltPath
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import PIL
from skimage import measure
import time
import argparse

'''
    Load tiff (magnetic field maps) and shape files (copper deposits GT polygons) and 
    create image np arrays and GT positive/negative masks of presence/absence of copper deposits.
    Run offline.
    Example:
    python GIA/utils/make_GT_masks.py --save_dir /home/dlserver/Documents/data/GIA/experimental --tif_dir /home/dlserver/Documents/data/GIA/benchmark/4010_data_set/4010_data_set/data_4010_igrf/DATA_4010_IGRF --shapefile_path /home/dlserver/Documents/data/GIA/benchmark/4010_data_set/4010_data_set/Delineations/20190618_4010_depos.shp --filename TMI_dX_CW
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


def explode_polygon(indata):
    indf = indata
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row] * recs, ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf, ignore_index=True)
    return outdf


def make_GT_masks(args=None):
    t_i = time.time()
    if args is None:
        args = getArgs()
    save_dir = args.save_dir
    filename = args.filename
    im_path = os.path.join(args.tif_dir, '{}.tif'.format(filename))
    print('\n\n ############################# {} #################################'.format(filename))
    shapefile_path = args.shapefile_path

    ####################### read a tif image ######################################
    ds = gdal.Open(im_path)
    im_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    print('tif file: {}'.format(im_arr.shape))
    mask_pos = np.zeros_like(im_arr)
    # im = Image.fromarray(im_arr)
    # im.show()
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
    nonblank_points_np = np.array(nonblank_points).transpose(1, 0)
    print('non-blank points: {}, {}'.format(len(nonblank_points[0]), len(nonblank_points[1])))

    # get width, height, min, max of a tif file in pixels and meters
    with rio.open(im_path) as tif_obj:
        extr_list = tif_obj.bounds
    print(tif_obj.meta)
    width_pixel = tif_obj.width
    height_pixel = tif_obj.height
    print('tif_obj height (pixels) = {}, width = {}'.format(height_pixel, width_pixel))
    x_min_m = extr_list[0]     # in meters
    x_max_m = extr_list[2]
    y_min_m = extr_list[1]
    y_max_m = extr_list[3]
    width_meter = x_max_m - x_min_m
    height_meter = y_max_m - y_min_m
    print('tif_file (meters): x_min_m = {}, x_max_m = {}, y_min_m = {}, y_max_m = {}, height = {}, width = {}'.
          format(x_min_m, x_max_m, y_min_m, y_max_m, height_meter, width_meter))

    # find connected components and remove them from negative GT mask
    labeled_mask_neg, n_labels = measure.label((1-mask_neg_with_blank), return_num=True, connectivity=2, background=False)  # (binary np array) gt_image shape is (H, W, 1)
    print('found {} connected components'.format(n_labels))
    mask_neg_without_blank_path = os.path.join(save_dir, 'mask_neg_without_blank_{}.npz'.format(filename))
    if not os.path.isfile(mask_neg_without_blank_path):
        mask_neg = np.zeros_like(labeled_mask_neg)
        for i in range(len(labeled_mask_neg)):
            mask_neg[i] = (np.abs(labeled_mask_neg[i] - 1)) > 0
            mask_neg = (mask_neg.astype(np.uint8))
        np.savez(mask_neg_without_blank_path, mask_neg)
        print('saved mask_neg without blank: {}'.format(mask_neg.shape))
    else:
        mask_neg = np.load(mask_neg_without_blank_path)['arr_0']
        print('loaded mask_neg without blank: {}'.format(mask_neg.shape))

    # calculate pixel size
    x_pixel_size = width_meter/(width_pixel-1)
    y_pixel_size = height_meter/(height_pixel-1)
    print('x_pixel_size = {}, y_pixel_size = {}'.format(x_pixel_size, y_pixel_size))

    # create interval of points in meters
    x = np.linspace(x_min_m, x_max_m, num=width_pixel)
    y = np.linspace(y_min_m, y_max_m, num=height_pixel)
    # print('x = {}'.format(x))
    # print('y = {}'.format(y))
    # print('len(x) = {}, len(y) = {}'.format(len(x), len(y)))

    # add all non-blank points coords (in meters) to a list
    # nonblank_points_m = [(x_min_m + nonblank_points_np[i, 1] * x_pixel_size, y_min_m + (height_pixel - nonblank_points_np[i, 0]) * y_pixel_size) for i in range(10000000)] #range(10000000)]
    # nonblank_points_m = [(x_min_m + nonblank_points_np[i, 1] * x_pixel_size, y_min_m + (height_pixel - nonblank_points_np[i, 0]) * y_pixel_size) for i in range(len(nonblank_points_np))]
    nonblank_points_m = []
    for i in range(len(nonblank_points_np)):
        nonblank_points_m.append((x_min_m + nonblank_points_np[i, 1] * x_pixel_size, y_min_m + (height_pixel - nonblank_points_np[i, 0]) * y_pixel_size))
    print('nonblank_points_m: {}'.format(len(nonblank_points_m)))


    #########################  read a shape file  ######################################
    # gdf = gpd.read_file(shapefile_path)
    # print('shape file: {}'.format(gdf.shape))
    # # gdf.plot()
    # # f, ax = plt.subplots(1)
    # # gdf.plot(ax=ax)
    # # plt.show()
    shape_file = ogr.Open(shapefile_path)
    source_layer = shape_file.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    print('shape_file: x_min = {}, x_max = {}, y_min = {}, y_max = {}'.format(x_min, x_max, y_min, y_max))

    # explode the deposits MultiPolygon into its constituents
    shapefile = gpd.read_file(shapefile_path)
    deposits = explode_polygon(shapefile)

    # nonblank_points_m = np.array([(505898.0, 7521699.0), (505841.0, 7521647.0), (505841.9, 7521647.1), (505898.01, 7521699.02)])
    #Loop over each individual polygon and get external coordinates
    for index, polygon in deposits.iterrows():
        mypolygon = []
        print('polygon idx = {}'.format(index))
        for pt in list(polygon['geometry'].exterior.coords):
            # print(index,', ',pt)
            mypolygon.append(pt)

        #See if any of the original grid points lie within the exterior coordinates of this polygon
        path = mpltPath.Path(mypolygon)
        inside = np.array(path.contains_points(nonblank_points_m))        # bool array of shape of 'nonblank_points_np'

        #find the results in the array that were inside the polygon ('True') and set them to missing
        idx_arr = np.where(inside==True)[0]                           # arrays of 'nonblank_points_np' idxs with deposits
        print('idx_arr: {}'.format(len(idx_arr)))
        for i in range(len(idx_arr)):
            idx = idx_arr[i]
            idx_i, idx_j = nonblank_points_np[idx]
            mask_pos[idx_i, idx_j] = index                         # mask of deposits (polygon idx)
            mask_neg[idx_i, idx_j] = 0                             # mask of no deposits (polygon idx)
    print('finished checking for points inside all polygons')

    # # neg and pos (with permutations) images
    im_arr_neg = np.zeros_like(im_arr)
    im_arr_pos = np.zeros_like(im_arr)
    for i in range(len(im_arr)):
        im_arr_neg[i] = np.multiply(im_arr[i], (mask_neg[i]>0)*1)
        im_arr_pos[i] = np.multiply(im_arr[i], (mask_pos[i]>0)*1)

    # save masks and images
    im_neg = Image.fromarray(im_arr_neg)
    im_neg = im_neg.resize((width_pixel // 10, height_pixel // 10), PIL.Image.ANTIALIAS)
    im_neg.save(os.path.join(save_dir, 'im_neg_{}.png'.format(filename)))
    im_pos = Image.fromarray(im_arr_pos)
    im_pos = im_pos.resize((width_pixel // 10, height_pixel // 10), PIL.Image.ANTIALIAS)
    im_pos.save(os.path.join(save_dir, 'im_pos_{}.png'.format(filename)))
    im_mask_neg_with_blank = Image.fromarray(mask_neg_with_blank*200)
    im_mask_neg_with_blank = im_mask_neg_with_blank.resize((width_pixel // 10, height_pixel // 10), PIL.Image.ANTIALIAS)
    im_mask_neg_with_blank.save(os.path.join(save_dir, 'im_mask_neg_with_blank_{}.png'.format(filename)))
    im_mask_neg = Image.fromarray(mask_neg * 200)
    im_mask_neg = im_mask_neg.resize((width_pixel//10, height_pixel//10), PIL.Image.ANTIALIAS)
    im_mask_neg.save(os.path.join(save_dir, 'im_mask_neg_{}.png'.format(filename)))
    im_mask_pos = Image.fromarray(mask_pos * 25)
    im_mask_pos = im_mask_pos.resize((width_pixel//10, height_pixel//10), PIL.Image.ANTIALIAS)
    im_mask_pos.save(os.path.join(save_dir, 'im_mask_pos_{}.png'.format(filename)))
    np.savez(os.path.join(save_dir, 'mask_neg_{}.npz'.format(filename)), mask_neg)
    np.savez(os.path.join(save_dir, 'mask_pos_{}.npz'.format(filename)), mask_pos)
    np.savez(os.path.join(save_dir, 'im_arr_{}.npz'.format(filename)), im_arr)
    np.savez(os.path.join(save_dir, 'im_arr_neg_{}.npz'.format(filename)), im_arr_neg)
    np.savez(os.path.join(save_dir, 'im_arr_pos_{}.npz'.format(filename)), im_arr_pos)

    print('run time: {} min'.format((time.time() - t_i)/60))



if __name__ == '__main__':
    make_GT_masks()

    # # read masks from file
    # mask_neg = np.load(os.path.join(save_dir, 'mask_neg.npz'))
    # mask_pos = np.load(os.path.join(save_dir, 'mask_pos.npz'))
    # im_arr = np.load(os.path.join(save_dir, 'im_arr.npz'))
    # print('mask_neg: {} {}'.format(mask_neg.files, mask_neg['arr_0'].shape))
    # print('mask_pos: {} {}'.format(mask_pos.files, mask_pos['arr_0'].shape))
    # print('im_arr: {} {}'.format(mask_pos.files, im_arr['arr_0'].shape))























