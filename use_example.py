from digits_classification import loadModel, predictNumber
import matplotlib.pyplot as plt



###############################################
img = plt.imread('data/images/6/6,0.jpg')
from digits_classification import normalizeSize
img = normalizeSize(img)
###############################################

classifier = loadModel('model.pkl')


number = predictNumber(img, classifier)
print(number)
