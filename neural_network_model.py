from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, \
    GlobalAveragePooling2D, Dropout

import keras
from keras import backend as K
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


# --------------------- create custom metric evaluation ---------------------
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# --------------------------- create model -------------------------------
def nn_model(max_len):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), name='conv0', padding='same', activation='relu',
                     input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation="sigmoid"))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model


def nn_model_new():
    # Define the input layer
    input_layer = keras.layers.Input(shape=(64, 64, 3))
    model = Sequential()

    #  VGG16

    # the first block of convolutional and max pooling layers
    model.add(Conv2D(filters=64, kernel_size=(3, 3), name='conv0', padding='same', activation='relu',
                     input_shape=(64, 64, 3)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # the second block of convolutional and max pooling layers
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # the third block of convolutional and max pooling layers
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # the fourth block of convolutional and max pooling layers
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # the fifth block of convolutional and max pooling layers
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation="sigmoid"))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model

# ------------------------- check model -----------------------------

checkpoint_path = "best_model.h5"

mc = ModelCheckpoint(filepath=checkpoint_path,
                     monitor='accuracy',
                     verbose=1,
                     save_best_only=True)

es = EarlyStopping(monitor='accuracy',
                   min_delta=0.01,
                   patience=20,
                   verbose=1)

cb = [mc]


def check_model(model_, x, y, x_val, y_val, epochs_, batch_size_):
    hist = model_.fit(x,
                      y,
                      epochs=epochs_,
                      batch_size=batch_size_,
                      validation_data=(x_val, y_val),
                      callbacks=cb
                      )

    model_.save("my_h5_model_backup.h5")

    return hist


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', embeddings_freq=0,
    embeddings_metadata=None
)

lr_callback = [lr_schedule, tb_callback]

accuracy = 0.91


class func_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= accuracy:
            print("Akurasi Telah Mencapai %2.2f%% , Proses Training Dihentikan." % (accuracy * 100))


callback = func_callback()


def check_model_new(model_, train_data, val_data, epochs_, batch_size):
    history = model_.fit(
        train_data,
        steps_per_epoch=623 // batch_size,
        epochs=epochs_,
        validation_data=val_data,
        validation_steps=155 // batch_size,
        verbose=2,
        callbacks=[lr_schedule, callback])

    model_.save("cervix_model_test.h5")

    return history


def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'],
             ['loss', 'val_loss'],
             ['precision', 'val_precision'],
             ['recall', 'val_recall']]
    for name in names:
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()


def evaluate_new_model(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # Mengambil Nilai Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # Plot Accruracy
    plt.plot(epochs, acc, 'r', label='Train accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    # Plot Loss
    plt.plot(epochs, loss, 'r', label='Train loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
