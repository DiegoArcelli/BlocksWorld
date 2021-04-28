import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import Sequential

BATCH_SIZE = 64
EPOCHS = 5

# immportazione del modello
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# train_mask = np.array([True for x in range(len(y_train))])
# test_mask = np.array([True for x in range(len(y_test))])

# for i in range(len(y_train)):
#     if y_train[i] > 6 or y_train[i] == 0:
#         train_mask[i] = False

# for i in range(len(y_test)):
#     if y_test[i] > 6 or y_test[i] == 0:
#         test_mask[i] = False

# x_train = x_train[train_mask]
# y_train = y_train[train_mask]
# x_test = x_test[test_mask]
# y_test = y_test[test_mask]

# for i in range(len(y_train)):
#     y_train[i] -= 1

# for i in range(len(y_test)):
#     y_test[i] -= 1


print(x_train[0].shape)

x_train = np.expand_dims(x_train, -1)
x_train = x_train / 255
x_test = np.expand_dims(x_test, -1)
x_test = x_test / 255


print(x_train[0].shape)

# building the model
model = Sequential()
model.add(Conv2D(filters=24, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(2, 2), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

print(model.predict(x_train[[0]]))

model.summary()

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

model.save("./model/model.h5")
