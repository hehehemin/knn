from data_utils import load_CIFAR10
from classifiers import KNearestNeighbor
import random
import numpy as np
import matplotlib.pyplot as plt

# set plt params
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = 'F:\研究生\李飞飞计算机视觉\CIFAR\cifar-10-python\cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
print
'x_train : ', x_train.shape
print
'y_train : ', y_train.shape
print
'x_test : ', x_test.shape, 'y_test : ', y_test.shape

# visual training example
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    # flaznonzero return indices_array of the none-zero elements
    # ten classes, y_train and y_test all in [1...10]
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        # subplot(m,n,p)
        # m : length of subplot
        # n : width of subplot
        # p : location of subplot
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(x_train[idx].astype('uint8'))
        # hidden the axis info
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# subsample data for more dfficient code execution
num_training = 5000
# range(5)=[0,1,2,3,4]
mask = range(num_training)
x_train = x_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = y_test[mask]
# the image data has three chanels
# the next two step shape the image size 32*32*3 to 3072*1
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print
'after subsample and re shape:'
print
'x_train : ', x_train.shape, " x_test : ", x_test.shape
# KNN classifier
classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
# compute the distance between test_data and train_data
dists = classifier.compute_distances_no_loops(x_test)
# each row is a single test example and its distances to training example
print
'dist shape : ', dists.shape
plt.imshow(dists, interpolation='none')
plt.show()
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
acc = float(num_correct) / num_test
print
'k=5 ,The Accurancy is : ', acc

# Cross-Validation

# 5-fold cross validation split the training data to 5 pieces
num_folds = 5
# k is params of knn
k_choice = [1, 5, 8, 11, 15, 18, 20, 50, 100]
x_train_folds = []
y_train_folds = []
x_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_acc = {}

for k in k_choice:
    k_to_acc[k] = []
for k in k_choice:
    print
    'cross validation : k = ', k
    for j in range(num_folds):
        # vstack :stack the array to matrix
        # vertical
        x_train_cv = np.vstack(x_train_folds[0:j] + x_train_folds[j + 1:])
        x_test_cv = x_train_folds[j]

        # >>> a = np.array((1,2,3))
        # >>> b = np.array((2,3,4))
        # >>> np.hstack((a,b))
        # horizontally
        y_train_cv = np.hstack(y_train_folds[0:j] + y_train_folds[j + 1:])
        y_test_cv = y_train_folds[j]

        classifier.train(x_train_cv, y_train_cv)
        dists_cv = classifier.compute_distances_no_loops(x_test_cv)
        y_test_pred = classifier.predict_labels(dists_cv, k)
        num_correct = np.sum(y_test_pred == y_test_cv)
        acc = float(num_correct) / num_test
        k_to_acc[k].append(acc)
print(k_to_acc)
