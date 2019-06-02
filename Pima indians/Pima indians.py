from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
"""Fully connected with 3 layers, 8 inputs"""

#Random numbers generator to ensure our results are reproducible and loaded data


dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter = ",")
#Data set has 9 columns
X = dataset[:,0:8]
Y = dataset [:,8]
"""Sequential model are defined by Dense class."""
model = tf.keras.models.Sequential([
    Dense(12, input_dim=(8), activation = tf.nn.relu),
    Dense(8, activation = tf.nn.relu),
    Dense(1, activation = tf.nn.sigmoid)
])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("/n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
