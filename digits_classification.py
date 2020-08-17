import numpy as np
import matplotlib.pyplot as plt
import pickle


def segmentation(img, threshold=0.1):
    img = img/255
    img_gray = ((img**2).sum(axis=2)**0.5)/(3**0.5)
    return img_gray  > (1 - threshold)


def normalizeSize(img, normal_size=[19, 13]):
    return img[1:normal_size[0]+1, 1:normal_size[1]+1]


def saveModel(model, path):
    with open(path, 'wb') as  f:
        f.write(pickle.dumps(model)) 


def loadModel(path):
    with open(path, 'rb') as f:
        model = pickle.loads(f.read())
    return model


def predictNumber(img, model):
    bw_img = segmentation(img)
    flat_img = bw_img.reshape(1, -1)
    prediction = model.predict(flat_img)[0]
    return prediction
