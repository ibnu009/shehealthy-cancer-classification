import cv2
import numpy
import numpy as np
from fastapi import FastAPI, File, UploadFile

from calculate_glcm_algo import calculate_glcm
from glcm_feature_extraction import glcm_feature_extraction
from cnn.read_image_file import read_imagefile, predict
from knn.knn_cervix_classification import knn

app = FastAPI()

@app.post("/classification_cnn/")
async def create_upload_file_cnn(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"message": "Image must be jpg or png format!", "data": '', "assumption": ''}

    contents = await file.read()
    image = read_imagefile(contents)
    opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    prediction = predict(image)
    result = glcm_feature_extraction(opencv_image)

    return {"message": "Berhasil", "data": result, "assumption": prediction}

@app.post("/classification_knn/")
async def create_upload_file_knn(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"message": "Image must be jpg or png format!", "data": '', "assumption": ''}

    contents = await file.read()
    image = read_imagefile(contents)
    opencv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    result = glcm_feature_extraction(opencv_image)
    features = calculate_glcm(gray)

    features = np.array(features)
    features = features.reshape(1, -1)

    print("Properties hasil reshape adalah ", features)

    pred = knn.predict(features)

    return {"message": "Berhasil", "data": result, "assumption": pred[0]}
