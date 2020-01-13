# Binary classification

#To solve: ValueError: Object arrays cannot be loaded when allow_pickle=False
import numpy as np
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# Load the imdb database
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

np.load = np_load_old

print(train_data[21])
print(train_labels[21])
print(len(train_data[21]))
print(train_data.shape)
print(train_data.dtype)
print(len(train_labels))


# encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, count in enumerate(sequences):
        results[i, count] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[21])


# vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# network architecture (model definition)
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compiling the model with optimizer, losses, and metrics configuration
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# training the model
history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# plotting the loss and accuracy values
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, 21)

plt.plot(epochs, loss_values, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.plot(epochs, acc_values, 'bo', label='training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Evaluate
results = model.evaluate(x_test, y_test)
print(results)

# Predict
y_pred = model.predict(x_test)
print(y_pred)

























