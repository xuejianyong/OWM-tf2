import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
import numpy as np

"""
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)
"""

DATASET_SIZE = 70000
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train)
print(x_test.shape)
print(y_test)

X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=((TEST_RATIO/(VALIDATION_RATIO+TEST_RATIO))))

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

print('--------------------------')
print(X_train.shape[3])









def split_mnist(mnist, cond):
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(mnist, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        sets_list.append(DataSet(this_set.images[cond(maxlabels),:], this_set.labels[cond(maxlabels)],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])

def test_lambda(mnist, condition):
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(mnist, set_name)



# dataset04 = test_lambda(mnist, lambda x: x < 5)

