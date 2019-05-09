from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
# import numpy as np

imdb = tf.keras.datasets.imdb

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000) #top 10 000 most frequently occuring words in data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)
#Convert the integers back to words

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
#The first indices are reserved
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)for (key, value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#post padding for converting to tensors
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding="post", maxlen=256)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding="post", maxlen=256)

#input shape is the vocabulary count used for the movie reviews
number_words = 10000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(number_words, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["acc"]) #binary crossentropy measures the distance beetween propability distributions

#Creating a validadion set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
patrial_y_train = y_train[10000:]

#Train the model
history = model.fit(partial_x_train, patrial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

#Evaluate the model
results = model.evaluate(x_test, y_test)
print(results)

#Graph of accuracy and loss

hist_dict = history.history
hist_dict.keys()

acc = hist_dict["acc"]
val_acc = hist_dict["val_acc"]
loss = hist_dict["loss"]
val_loss = hist_dict["val_loss"]

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

