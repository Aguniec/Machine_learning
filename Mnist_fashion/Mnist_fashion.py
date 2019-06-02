import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, x_test = x_train / 255.0, x_test / 255.0

"""plt.figure(figsize=(10,10))
for i in range (25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names [y_train[i]])
plt.show()
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), #transform format from 2-D array to 1-D array x*y
    keras.layers.Dense(128, activation = tf.nn.relu), #fully-connected, 128 neurons
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
#optimizer - how model is updated base on the data it sees and loss function
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#Traning model
model.fit(x_train, y_train, epochs = 5)

#Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

#Make predictions
predictions = model.predict(x_test)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array),class_names[true_label]), color = color)

def plot_value_array (i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thispot = plt.bar(range(10), predictions_array, color = "yellow")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thispot[predicted_label].set_color("red")
    thispot[true_label].set_color("blue")

"""
num_rows = 5
num_cols = 3
num_images = num_cols*num_rows

plt.figure(figsize= (2*2*num_cols, 2*num_rows))
for i in range (num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i +1)
    plot_image(i, predictions, y_test, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i +2)
    plot_value_array(i, predictions, y_test)
plt.show()
"""
#Make a predicton for a single image
img = x_test[0]
img = np.expand_dims(img,0)

predictions_single = model.predict(img)

plot_value_array(0, predictions_single, y_test)
_= plt.xticks(range(10), class_names, rotation = 45)
plt.show()

np.argmax(predictions_single[0])


