import cv2 as cv
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.special import softmax

# importing the training and test sets stored as numpy's arrays

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

train_mask = np.array([True for x in range(len(y_train))])
test_mask = np.array([True for x in range(len(y_test))])

for i in range(len(y_train)):
    if y_train[i] > 6 or y_train[i] == 0:
        train_mask[i] = False

for i in range(len(y_test)):
    if y_test[i] > 6 or y_test[i] == 0:
        test_mask[i] = False

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

for i in range(len(y_train)):
    y_train[i] -= 1

for i in range(len(y_test)):
    y_test[i] -= 1

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# training set of 60000 images
print("Training set shape:")
print(x_train.shape)
print(y_train.shape)

# test set of 10000 images
print("Test set shape:")
print(x_test.shape)
print(y_test.shape)


# every element of is a 28x28 matrix that rappresent a digit from 0 to 9
# every element of the matrix ranges from 0 to 255, where the value
# rappresent the grey scale

print(np.max(x_train))
print(np.min(x_train))

# reshape the  matricies into a one dimensional vector of 28*28 elements
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

print(x_train.shape)
print(x_test.shape)


# normalization of the values of the pixels of the images

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print(np.max(x_train))
print(np.min(x_train))

n_classes = 6

# using a perceprton as model
# TO DO: add more layers
model = keras.Sequential()

# model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(512, activation="relu", input_shape=(784,)))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dropout(0.3))
model.add(layers.Dense(6, activation="softmax"))


batch_size = 256
epochs = 30
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print(model.predict(x_test[:1]))

model.save("./model/model.h5")


# new_model = keras.models.load_model("./model/model.h5")
# print(new_model.predict(x_test[:1]))
# img = cv.imread("mie/1b.jpg", cv.CV_8U)
# # img_gauss = cv.GaussianBlur(img, (5, 5), 0)
# ret, thrash = cv.threshold(img, 60, 255, cv.THRESH_BINARY)
# thrash = ~thrash
# thrash = cv.resize(thrash, (28, 28), interpolation=cv.INTER_AREA)

# thrash = thrash.reshape(1, 784)
# print(new_model.predict(thrash[:1]))
