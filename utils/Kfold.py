import numpy as np
from sklearn.model_selection import KFold

root = "/Users/andreanascetti/PycharmProjects/Landslide/dataset/"
path_test = "test.txt"
path_train = "trainval.txt"

with open(root + path_test) as f:
    files = f.readlines()

with open(root + path_train) as f:
    base_files = f.readlines()

kf = KFold(n_splits=3, shuffle=True)

id =1

array_files = np.array(files)

for train_index, test_index in kf.split(files):

    # print("TRAIN:", train_index, "TEST:", test_index)
    with open(root + "_Kfold_"+str(id)+"_train.txt", "w") as f:
        f.writelines(base_files)
        f.writelines(array_files[train_index])

    with open(root + "_Kfold_"+str(id)+"_valid.txt", "w") as f:
        f.writelines(array_files[test_index])

    id+= 1