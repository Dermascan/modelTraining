from skimage import feature
import cv2
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('hmnist_28_28_RGB.csv')

# Shuffle the dataframe
df = df.sample(frac=1)

# Pre Process Pixel Data
def preprocess(img):

    img = cv2.convertScaleAbs(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Grayscale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(img_gray, 100, 250)

    # Local Binary Pattern
    pattern_img = cv2.equalizeHist(img_gray)
    pattern_img = feature.local_binary_pattern(pattern_img, P=4, R=2, method='uniform')
    pattern_img = pattern_img.astype(np.uint8)
    pattern_img = cv2.resize(pattern_img, (28, 28))

    # Concatenate all channels
    result_img = np.dstack([img, edges, pattern_img])
    return result_img

train_set, test_set = train_test_split(df, test_size=0.2)


y_train = train_set['label']
x_train = train_set.drop(columns=['label'])

y_test = test_set['label']
x_test = test_set.drop(columns=['label'])


oversample = RandomOverSampler()
x_train, y_train = oversample.fit_resample(x_train, y_train)
x_train = np.array(x_train).reshape(-1, 28, 28, 3)
x_test = np.array(x_test).reshape(-1, 28, 28, 3)
x_train = np.array([preprocess(img) for img in x_train])
x_test = np.array([preprocess(img) for img in x_test])


from keras_tuner.tuners import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv1_filters', min_value=8, max_value=32, step=8),
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        input_shape=(28, 28, 5)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv2_filters', min_value=16, max_value=64, step=16),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv3_filters', min_value=32, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv4_filters', min_value=64, max_value=256, step=64),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv5_filters', min_value=128, max_value=512, step=128),
        kernel_size=(3, 3),
        activation='relu'
    ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense1_units', min_value=64, max_value=512, step=64),
        activation='relu'
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense2_units', min_value=32, max_value=256, step=32),
        activation='relu'
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense3_units', min_value=16, max_value=128, step=16),
        activation='relu'
    ))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
tuner = kt.Hyperband(
    build_model,
    objective='accuracy',
    max_epochs=80,
    factor=3,
    directory='HPT',
    project_name='v8.2'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

log_dir = 'logs/v8.2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner.search(
    x_train, 
    y_train, 
    epochs=80, 
    validation_data=(x_test, y_test),
    callbacks=[stop_early, tensorboard_callback])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('modelv8_2.keras')