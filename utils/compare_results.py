import os
import glob
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np

image_dir = "/Volumes/GeomaticsProjects1/Projects/Satellite_Photogrammetry/LandSlide/TestData/img/"
root = "/Users/andreanascetti/PycharmProjects/Landslide/"
directory = root + "test_results_DeepLabPlus_retrained/predictions/batch10000/" #"/test_results_DeepLabPlus_pretrained9500/"
results = root + "test_results_DeepLabPlus_retrained2/predictions/batch8000/"
#results = "/Volumes/GeomaticsProjects1/Projects/Satellite_Photogrammetry/LandSlide/ValidationData/submissions/CE_L4S_sup_selection_CM/"
files = glob.glob(directory+"/*.h5")
print(files)

f1s = []
slopes =[]

for f in files:
    hdf = h5py.File(f, 'r')
    hdf_pred = h5py.File(results + f.split('/')[-1], 'r') # .replace('mask', 'image')
    array_mask = hdf["mask"][:].squeeze()
    array_pred = hdf_pred["mask"][:].squeeze()
    f1 = f1_score(array_mask.astype('uint8').flatten(), array_pred.astype('uint8').flatten())
    print(f1)
    #print("mean_slope: ", np.mean(slope_masked))
    #print("mean_slope: ", np.mean(slope_masked[np.where(slope_masked > 0)]))
    f1s.append(f1)
    if f1 > 0.1:
        hdf_img = h5py.File(image_dir + f.split('/')[-1].replace('mask', 'image'), 'r')  #
        img = hdf_img["img"][:, :, 1]
        dsm = hdf_img["img"][:, :, 13].squeeze()
        slope = hdf_img["img"][:, :, 12].squeeze()
        slope_masked = slope * array_mask
        slopes.append(np.mean(slope_masked[np.where(slope_masked > 0)]))
    #print(img.shape)
        #hillshade = es.hillshade(dsm)

        #f, axarr = plt.subplots(2, 2)
        #axarr[0, 0].imshow(array_mask.astype('uint8'))
        #axarr[1, 0].imshow(array_pred.astype('uint8'))
        #axarr[0, 1].imshow(img)
        #axarr[1, 1].imshow(dsm)
        #plt.show()

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ep.plot_bands(
        #     dsm,
        #     ax=ax,
        #     cmap="terrain",
        #     title="Lidar Digital Elevation Model (DEM)\n overlayed on top of a hillshade",
        # )
        # ax.imshow(hillshade, cmap="Greys", alpha=0.5)
        # plt.show()

plt.hist(f1s, bins='auto')  # arguments are passed to np.histogram
plt.title("F1")
plt.show()

plt.hist(slopes, bins='auto')  # arguments are passed to np.histogram
plt.title("Slopes")
plt.show()

f1s = np.array(f1s)
print("Results")
print(np.mean(f1s))
print(np.std(f1s))
print(np.mean(slopes))
print(np.std(slopes))


    #check the slope values
