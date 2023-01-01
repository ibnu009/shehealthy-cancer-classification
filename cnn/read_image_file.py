from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from numpy import array

class_predictions = array([
    'NORMAL',
    'PRECANCER',
])


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def predict(image: Image.Image):
    model = keras.models.load_model("./cnn/my_h5_model_backup.h5")

    image = np.asarray(image.resize((64, 64)))[..., :3]

    img_array = tf.keras.utils.img_to_array(image)

    img_array = img_array / 255.

    # result = decode_predictions(model.predict(img_array), 2)[0]
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    result = model.predict(img_array)

    score = tf.nn.softmax(result[0])

    print("pred result is ", score)
    print(
        "This image most likely belongs to {}"
        .format(class_predictions[np.argmax(score)])
    )

    return class_predictions[np.argmax(score)]
