import cv2
import numpy as np
from calculate_glcm_algo import calculate_glcm
from knn.knn_cervix_classification import knn

img = cv2.imread("../../img_she_healthy/testing/normal1.jpg")
# img = cv2.imread("./precancer2.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


features = calculate_glcm(gray)
print("Properties adalah ", features)

features = np.array(features)
features = features.reshape(1, -1)

print("Properties hasil reshape adalah ", features)

pred = knn.predict(features)

print("Hasil prediksi adalah ", pred)
