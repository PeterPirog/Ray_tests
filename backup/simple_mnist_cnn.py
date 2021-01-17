import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

print('Is cuda available:', tf.test.is_gpu_available())
batch_size = 128
num_classes = 10
epochs = 200

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test,axis=3)


IMG_SHAPE = x_train.shape
print('x_train shape=', x_train.shape)
print('x_test shape=', x_test.shape)




# define model
inputs = tf.keras.layers.Input(shape=(28,28,1))  # changed size shape=(28, 28)
# 1st conv layer
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D((2, 2))(x)
# 2nd conv layer
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu')(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=245, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1211)(x)
outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_conv_model")
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=0.00148),
    metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

