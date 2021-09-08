"""
QUESTION

Build and train a neural network to predict stock prices of Apple

Network will be scored based on the MAE you have.
"""

import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from urllib.request import urlretrieve



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    url = "https://raw.githubusercontent.com/kyleconroy/apple-stock/master/apple_stock_data.csv"
    urlretrieve(url, 'apple_stocks.csv')

    time_step = []
    close = []

    with open("apple_stocks.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            close.append(float(row[-1]))
            time_step.append(row[0])

    series = pd.Series(close, index=time_step)


    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # Split data on the 1500th element

    split_time = 4500
    time_train = time[split_time]  # code
    x_train = series[time_train:]  # Code
    time_valid = time[split_time + 1]  # code
    x_valid = series[:time_valid]  # code


    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size,
                                 batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)

    valid_set = windowed_dataset(x_valid, window_size=window_size,
                                 batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([

        tf.keras.layers.LSTM(100, input_shape=(None, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
    model.fit(train_set, epochs=10, validation_data=valid_set)

    return model


#save your model in the .h5
if __name__ == "__main__":
    model = solution_model()
    model.save("apple.h5")

