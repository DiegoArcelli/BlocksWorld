import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import Sequential

# file per allenare e salvare la rete neurale che effettua il riconoscimento delle cifre
# il modello viene allenato sul dataset del MNIST

BATCH_SIZE = 64
EPOCHS = 10

# si estraggono e si 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# si aggiunge la dimensione del canale e si normalizza il valore dei pixel tra 0 e 1
x_train = np.expand_dims(x_train, -1)
x_train = x_train / 255
x_test = np.expand_dims(x_test, -1)
x_test = x_test / 255

# definizione del modello
model = Sequential()
model.add(Conv2D(filters=24, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=(3, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.predict(x_train[[0]])

model.summary()

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# allenamento del modello
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

# calcolo della precisione e dell'errore nel validation set
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss', test_loss)
print('Test accuracy:', test_acc)

# plot dei grafici relativi all'andamento di accuracy e loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

model.save("./model/model.h5")
