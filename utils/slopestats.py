import os
import glob
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

image_dir = "/Volumes/GeomaticsProjects1/Projects/Satellite_Photogrammetry/LandSlide/TrainData/img/"
mask_dir = "/Volumes/GeomaticsProjects1/Projects/Satellite_Photogrammetry/LandSlide/TrainData/mask/"

files = glob.glob(image_dir+"/*.h5")

f1s = []
slopes_mean =[]
slopes_min = []
slopes_max = []
slopes_landslide =[]
slopes_landslide_min =[]

for f in files[0:50]:
    print(f)
    hdf_mask = h5py.File(mask_dir + f.split('/')[-1].replace('image', 'mask'), 'r')
    mask = hdf_mask["mask"][:].squeeze()

    if np.mean(mask) > 0:
        hdf_img = h5py.File(f, 'r')
        img = hdf_img["img"][:, :, 1]
        dsm = hdf_img["img"][:, :, 13].squeeze()
        slope = hdf_img["img"][:, :, 12].squeeze()
        slope_masked = slope * mask

        slopes_mean.append(np.mean(slope))
        slopes_min.append(np.min(slope))
        slopes_max.append(np.max(slope))
        slopes_landslide.append(np.mean(slope_masked[np.where(slope_masked > 0)]))
        slopes_landslide_min.append(np.min(slope_masked[np.where(slope_masked > 0)]))

plt.hist(slope, bins='auto')  # arguments are passed to np.histogram
plt.title("Mean_Tile_Slope")
plt.show()

plt.hist(slopes_min, bins='auto')  # arguments are passed to np.histogram
plt.title("Min_Tile_Slope")
plt.show()

plt.hist(slopes_landslide, bins='auto')  # arguments are passed to np.histogram
plt.title("Mean_Landslide_Slope")
plt.show()

plt.hist(slopes_landslide_min, bins='auto')  # arguments are passed to np.histogram
plt.title("Min_Landslide_Slope")
plt.show()
