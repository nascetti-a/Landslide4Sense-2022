import os
import glob
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
import richdem as rd

root = "/Users/andreanascetti/PycharmProjects/Landslide/"

image_dir = "/Volumes/GeomaticsProjects1/Projects/Satellite_Photogrammetry/LandSlide/TestData/img/"

model1_path = root + "test_results_DeepLabPlus_retrained_kFold/fold1/predictions/batch10000/"
model2_path = root + "test_results_DeepLabPlus_retrained_kFold/fold2/predictions/batch6000/"
model3_path = root + "test_results_DeepLabPlus_retrained_kFold/fold3/predictions/batch10000/"

files = glob.glob(model1_path+"/*.h5")

for f in files:

    hdf_mask1 = h5py.File(f, 'r')
    mask1 = hdf_mask1["mask"][:].squeeze()

    hdf_mask2 = h5py.File(model2_path + f.split('/')[-1], 'r')
    mask2 = hdf_mask2["mask"][:].squeeze()

    hdf_mask3 = h5py.File(model3_path + f.split('/')[-1], 'r')
    mask3 = hdf_mask3["mask"][:].squeeze()

    tot_mask = mask1+mask2+mask3
    color_list_20530 = ['blue', 'cyan', 'red', 'yellow']
    cmap_20530 = ListedColormap(color_list_20530)

    (unique, counts) = np.unique(tot_mask, return_counts=True)
    print(counts)
    if len(counts) == 4:
        if counts[2] > 150:

            hdf_img = h5py.File(image_dir + f.split('/')[-1].replace('mask', 'image'), 'r')
            img = hdf_img["img"][:, :, 1]
            dsm = hdf_img["img"][:, :, 13].squeeze()
            slope = hdf_img["img"][:, :, 12].squeeze()

            dim = (64, 64)

            dsm_resized = cv2.resize(dsm, dim, interpolation=cv2.INTER_AREA)

            accum_d8 = np.log(rd.FlowAccumulation(rd.rdarray(dsm_resized, no_data= -9999), method='D8'))
            print(np.max(accum_d8), np.min(accum_d8))
            aspect = rd.TerrainAttribute(rd.rdarray(dsm_resized, no_data= -9999), attrib='aspect')/360.0

            #raster = Raster(array, viewfinder=ViewFinder(shape=array.shape))

            f, axarr = plt.subplots(3, 2)
            axarr[0, 0].imshow(img)
            axarr[1, 0].imshow(dsm)
            axarr[0, 1].imshow(accum_d8)
            axarr[1, 1].imshow(aspect)
            axarr[2, 0].imshow(slope)
            axarr[2, 1].imshow(tot_mask, cmap=cmap_20530)
            plt.show()
