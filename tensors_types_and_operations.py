import numpy as np

# Scalars
x = np.array(71)
print(x)
print(x.ndim)

# Vectors
y = np.array([12, 3, 5, 733])
print(y)
print(y.ndim)

# Matrices
z = np.array([[6, 3, 5, 13],
             [7, 434, 5, 0.1],
             [5, 1, -1, -1.5]])
print(z)
print(z.ndim)

# 4D tensors
w = np.array([[[6, 3, 5, 13],
             [7, 434, 5, 0.1],
             [5, 1, -1, -1.5]],
             [[6, 3, 5, 13],
              [7, 434, 5, 0.1],
              [5, 1, -1, -1.5]],
             [[6, 3, 5, 13],
              [7, 434, 5, 0.1],
              [5, 1, -1, -1.5]]])
print(w)
print(w.ndim)

# Attributes
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[5]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice_1 = train_images[10:100]
print(my_slice_1.shape)

my_slice_2 = train_images[10:100, :, :]
print(my_slice_2.shape)

my_slice_3 = train_images[10:100, 0:28, 0:28]
print(my_slice_3.shape)

my_slice_4 = train_images[:, 14:, 14:]
print(my_slice_4.shape)

my_slice_5 = train_images[:, 7:-7, 7:-7]
print(my_slice_5.shape)

batch_1 = train_images[:128]
batch_2 = train_images[128:256]
n = 10
batch_n = train_images[128*n:128*(n+1)]


















