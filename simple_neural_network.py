import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy as np
import graphviz
import pydot

# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

print (data.shape)
print (labels.shape)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
