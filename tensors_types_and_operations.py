# scalar 0D tensor
import numpy as np

a = np.array(23)
print(a.ndim)

# vector 1D tensor
b = np.array([32, 2, 4, 1, 2342])
print(b, b.ndim)

# matrices 2D tensors
c = np.array([[2, 43, 53, 2],
              [4, 23, 6, 81],
              [9, 34, 76, 1]])
print(c, c.ndim)

# 4D tensors
d = np.array([[[2, 43, 53, 2],
               [4, 23, 6, 81],
               [9, 34, 76, 1]],
              [[2, 43, 53, 2],
               [4, 23, 6, 81],
               [9, 34, 76, 1]],
              [[2, 43, 53, 2],
               [4, 23, 6, 81],
               [9, 34, 76, 1]]])
print(d, d.ndim)

# attributes: shape, ndim, dtype
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_images.ndim)
print(train_images.dtype)

# display
import matplotlib.pyplot as plt

digit = train_images[5833]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# tensor slicing
slice_1 = train_images[200:220]
print(slice_1.shape)

slice_2 = train_images[200:220, :, :]
print(slice_2.shape)

slice_3 = train_images[200:220, 0:28, 0:28]
print(slice_3.shape)

slice_4 = train_images[:, 14:, 14:]
print(slice_4.shape)

slice_5 = train_images[:, 7:-7, 7:-7]

# Batch
batch_1 = train_images[:128]
batch_2 = train_images[128:256]
n = 10
batch_n = train_images[(n - 1) * 128:n * 128]


# tensor operations
def naive_relu(x):
    assert len(x.shape) == 2  # x must be a 2D Numpy tensor

    x = x.copy()  # not to overwrite the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(0, x[i, j])
    return x


e = np.random.random((5, 3)) - 0.5
e_relu = naive_relu(e)
print(e)
print(e_relu)


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


f = np.random.random((5, 3))
g = naive_add(e, f)
g_check = e + f
print(g)
print(g_check)

# Based on BLAS
import numpy as np

h = np.maximum(e, 0)  # element-wise relu
k = e + f  # element-wise addition


# Broadcasting
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            x[i, j] += y[j]
    return x


l = np.random.random((32, 10))
m = np.random.random(10, )
sum_broadcast = naive_add_matrix_and_vector(l, m)
sum_broadcast_check = l + m
print(sum_broadcast - sum_broadcast_check)

n = np.maximum(l, m)

# tensor dot
import numpy as np

a_for_dot = np.random.random((4, 3))
b_for_dot = np.random.random((3, 5))
c_dot_check = np.dot(a_for_dot, b_for_dot)


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


a_for_dot_vector = np.random.random(8, )
b_for_dot_vector = np.random.random(8, )
c_vector_dot = naive_vector_dot(a_for_dot_vector, b_for_dot_vector)
c_vector_dot_check = np.dot(a_for_dot_vector, b_for_dot_vector)
print(c_vector_dot)
print(c_vector_dot_check)


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


a_for_dot_matrix = np.random.random((6, 8))
c_matrix_vector_dot = naive_matrix_vector_dot(a_for_dot_matrix, b_for_dot_vector)
c_matrix_vector_dot_check = np.dot(a_for_dot_matrix, b_for_dot_vector)
print(c_matrix_vector_dot)
print(c_matrix_vector_dot_check)


def naive_matrix_vector_dot_2(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


c_matrix_vector_dot_2 = naive_matrix_vector_dot_2(a_for_dot_matrix, b_for_dot_vector)
print(c_matrix_vector_dot_2)
print(c_matrix_vector_dot_check)


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row = x[i, :]
            column = y[:, j]
            z[i, j] = naive_vector_dot(row, column)
    return z


b_for_dot_matrix = np.random.random((8, 5))
c_matrix_dot = naive_matrix_dot(a_for_dot_matrix, b_for_dot_matrix)
c_matrix_dot_check = np.dot(a_for_dot_matrix, b_for_dot_matrix)
print(c_matrix_dot - c_matrix_dot_check)


# tensor reshape
a_for_reshape = np.array([[3, 4],
                          [5, 6],
                          [7, 8]])
a_after_reshape_1 = a_for_reshape.reshape((6, 1))
a_after_reshape_2 = a_for_reshape.reshape((2, 3))

print(a_for_reshape.shape)
print(a_after_reshape_1)
print(a_after_reshape_2)

# transpose
a_for_transpose = np.zeros((400, 30))
a_after_transpose = np.transpose(a_for_transpose)
print(a_after_transpose.shape)













