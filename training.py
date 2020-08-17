import os, glob
import numpy as np
import matplotlib.pyplot as plt
from digits_classification import segmentation, normalizeSize
from digits_classification import loadModel, saveModel
from sklearn import svm


def loadNumberSet(number, folder):
    number_folder = os.path.join(folder, str(number))
    files = [os.path.join(number_folder, file) for file in os.listdir(number_folder) if file.endswith(".jpg")]
    images = [plt.imread(file) for file in files]
    bw_images = [segmentation(image) for image in images]
    clean_images = [normalizeSize(image) for image in bw_images]
    return clean_images


def flatImages(images):
    buffer = np.zeros([len(images), images[0].size], dtype=float)
    for i in range(len(images)):
        buffer[i] = images[i].reshape(-1)
    return buffer


def loadAllData(numbers = range(0, 9), folder =  os.path.join('data', 'images')):
    X = []
    y = []
    for number in numbers:
        number_data = loadNumberSet(number, folder)
        flat_images = flatImages(number_data)
        X.append(flat_images)
        y.append(np.zeros(len(flat_images), dtype=int) + number)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def shuffeData(X, y):
    index = np.array(range(len(X)), dtype = int)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y


X, y = loadAllData()
X, y = shuffeData(X, y)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.1)
classifier.fit(X, y)

saveModel(classifier, 'model.pkl')

classifier = loadModel('model.pkl')

print(classifier.score(X, y))
