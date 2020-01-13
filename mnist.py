# MNIST as the Hello World of DL

# Load the MNIST dataset in Keras
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)

# The network architecture, Sequential class
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# # The network architecture, functional API
# input_tensor = layers.Input(shape=(28*28,))
# x = layers.Dense(512, activation='relu')(input_tensor)
# output_tensor = layers.Dense(10, activation='softmax')(x)
# model = models.Model(input_tensor, output_tensor)

# Compilation
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Preparing the image data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# Preparing the labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fit
network.fit(train_images, train_labels, epochs=5, batch_size=128)   # 469 gradient update per epoch (60000/128)
# Evaluate
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# # optimization with momentum, naive implementation
# past_velocity = 0
# momentum = 0.1  # momentum factor
# while loss > 0.01:  # optimization loop
#     w, loss, gradient = get_current_parameters()
#     velocity = past_velocity * momentum + learning_rate * gradient
#     w = w - learning_rate * gradient + velocity * momentum
#     past_velocity = velocity
#     update_parameter(w)

























