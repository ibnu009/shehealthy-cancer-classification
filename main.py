import os
import re
import cv2
import pandas as pd

import numpy as np
import skimage
from skimage.feature import greycomatrix, greycoprops

input_matrix = np.array([[0, 0, 1],
                         [1, 2, 3],
                         [2, 3, 2]])

glcm = greycomatrix(input_matrix,
                    distances=[1],
                    angles=[0],
                    levels=4,
                    symmetric=True,
                    normed=True)

print(glcm[:, :, 0, 0])

# -------------------- Utility function ------------------------


def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("", "", "()"))
    str_ = str_.split("_")
    return ''.join(str_[:2])


def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    text = text.strip()
    return text


def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#" * round((val) * bar_size / val_len) + " " * round((val_len - (val)) * bar_size / val_len)
    if val == 0:
        print("", end="\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")

# -------------------- Load Dataset ------------------------


dataset_dir = "../img_she_healthy/DATASET"

imgs = []  # list image matrix
imgsx = []  # list image matrix

labels = []
descs = []

# Ngambil file
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))

        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resize = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

            imgs.append(resize)
            imgsx.append(gray)
            labels.append(sub_folder)
            descs.append(normalize_desc(folder, sub_folder))

            print_progress(i, len_sub_folder, folder, sub_folder, filename)

# ----------------- calculate GLCM for angle 0, 45, 90, 135 -----------------------
def calc_glcm_all_agls(img, label, props, dists=[1], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = greycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in skimage.feature.graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)

    return feature


# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
properties = ['contrast', 'dissimilarity', 'energy', 'correlation']
angles = ['0', '45', '90', '135']
columns = []
glcm_all_agls = []

for img, label in zip(imgs, labels):
    glcm_all_agls.append(
        calc_glcm_all_agls(img,
                           label,
                           props=properties)
    )

for name in properties:
    for ang in angles:
        columns.append(name + "_" + ang)


columns.append("label")

glcm_df = pd.DataFrame(glcm_all_agls,
                      columns = columns)

#save to csv
glcm_df.to_csv("glcm_cervix.csv")

glcm_df.head(7)



