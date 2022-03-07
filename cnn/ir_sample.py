#!/usr/bin/env python3

"""
ImageRecognition Sample

https://github.com/StackAbuse/sa-image-recognition-keras/blob/main/SA-ImageRecognition.ipynb
"""

import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10

# Set random seed for purposes of reproducibility
seed = 21

"""
load dataset
X: input data, training and testing
Y: corresponding output(classification), training and testing
"""
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
"""
normalize data: from 0~255 integer to 0~1 float
astype(): numpy method to convert data type
"""
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

"""
one-hot encode output: one bit indicate one category; easy to classify
"""
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
print("class number", class_num)

"""
Design CNN model
define the format we would like to use for the model, Keras has several different formats or blueprints to build models on, but Sequential is the most commonly used
"""
"""
Sequential groups a linear stack of layers into a tf.keras.Model.
Sequential provides training and inference features on this model.
"""
model = keras.Sequential()
"""
#1st layer: a convolutional layer. It will take in the inputs and run convolutional filters on them
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
# of channels/filters: 32 - does this have to be equal to image size or it's just a conincidence here?
"""
model.add(keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu', padding='same'))
"""
Applies Dropout to the input.
The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
Now we will add a dropout layer to prevent overfitting, which functions by randomly eliminating some of the connections between the layers (0.2 means it drops 20% of the existing connections)
"""
model.add(keras.layers.Dropout(0.2))
"""
Batch Normalization normalizes the inputs heading into the next layer, ensuring that the network always creates activations with the same distribution that we desire
"""
model.add(keras.layers.BatchNormalization())
"""
These are the basic blocks used for building CNNs:
    Convolutional layer
    activation
    dropout
    pooling
 These blocks can then be stacked, typically in an pyramid pattern in terms of complexity.
 The next block typically contains a convolutional layer with a larger filter, which allows it to find patterns in greater detail and abstract further,
 followed by a pooling layer, dropout and batch normalization
"""

#layer2
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

#layer3
model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

#layer4
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

"""
After we are done with the convolutional layers, we need to Flatten the data. We'll also add a layer of dropout again
"""
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

"""
Now we make use of the Dense import and create the first densely connected layer.
We need to specify the number of neurons in the dense layer.
Note that the numbers of neurons in succeeding layers decreases, eventually approaching the same number of neurons as there are classes in the dataset (in this case 10).

We can have multiple dense layers here, and these layers extract information from the feature maps to learn to classify images based on the feature maps.
Since we've got fairly small images condensed into fairly small feature maps - there's no need to have multiple dense layers. A single, simple, 32-neuron layer should be quite enough:
"""
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

"""
In the final layer, we pass in the number of classes for the number of neurons.
Each neuron represents a class, and the output of this layer will be a 10 neuron vector with each neuron storing some probability that the image in question belongs to the class it represents.

Finally, the softmax activation function selects the neuron with the highest probability as its output, voting that the image belongs to that class:
"""
model.add(keras.layers.Dense(class_num, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

numpy.random.seed(seed)
"""
will tf choose how to feed data in each epoch?
"""
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)

"""
Note that in most cases, you'd want to have a validation set that is different from the testing set, and so you'd specify a percentage of the training data to use as the validation set.
In this case, we'll just pass in the test data to make sure the test data is set aside and not trained on. We'll only have test data in this example, in order to keep things simple.
"""
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()
plt.show()
