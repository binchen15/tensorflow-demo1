"""tensorflow tutorial using fashion_mnist dataset"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

train, test = data.load_data()

train_images, train_labels = train
test_images, test_labels = test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# images size 28 * 28
# print(train_images[7])


# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_accuracy}")

predictions = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: {}".format(test_labels[i]))
    plt.title("Predicdtion {}".format(class_names[np.argmax(predictions[i])]))
    plt.show()






