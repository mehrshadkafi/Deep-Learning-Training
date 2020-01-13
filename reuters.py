# Multiclass classification
#To solve: ValueError: Object arrays cannot be loaded when allow_pickle=False
import numpy as np
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# Load the reuters database
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

np.load = np_load_old

print(train_data)
print(train_data.shape)
print(train_data[1243])
print(len(train_data[1243]))
print(train_labels)
print(train_labels[1243])


# Encoding the data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


one_hot_y_train_naive= to_one_hot(train_labels)
one_hot_y_test_naive = to_one_hot(test_labels)

from keras.utils.np_utils import to_categorical
one_hot_y_train = to_categorical(train_labels)
one_hot_y_test = to_categorical(test_labels)

# model definition
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# model compilation
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data for validation
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_y_train[:1000]
partial_y_train = one_hot_y_train[1000:]

# training
history = model.fit(partial_x_train, partial_y_train, epochs=9,
                    batch_size=512, validation_data=(x_val, y_val))

# plot
history_dict = history.history
history_dict.keys()
epochs = range(1, 10)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

import matplotlib.pyplot as plt
plt.plot(epochs, loss_values, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation losss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc_values, 'bo', label='training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# evaluate
results = model.evaluate(x_test, one_hot_y_test)
print(results)

# prediction
predictions = model.predict(x_test)
print(predictions[0])
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

# # different way of handling the labels
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
#
# model.compile(optimizer='rmsprop',
#               loss='sparse_categorical_crossentropy',
#               metrics=['acc'])


# bottleneck
model_bottleneck = models.Sequential()

model_bottleneck.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model_bottleneck.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model_bottleneck.add(layers.Dense(46, activation='softmax', input_shape=(10000,)))

model_bottleneck.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         metrics=['acc'])

model_bottleneck.fit(partial_x_train, partial_y_train, epochs=9,
                     batch_size=512, validation_data=(x_val, y_val))

results_bottleneck = model_bottleneck.evaluate(x_test, one_hot_y_test)
print(results_bottleneck)
print(results)

# excess weights
model_excess = models.Sequential()
model_excess.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model_excess.add(layers.Dense(128, activation='relu'))
model_excess.add(layers.Dense(46, activation='softmax'))

model_excess.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

model_excess.fit(partial_x_train, partial_y_train, epochs=9,
          batch_size=512, validation_data=(x_val, y_val))

results_excess = model_excess.evaluate(x_test, one_hot_y_test)
print(results_excess)
print(results)


































































