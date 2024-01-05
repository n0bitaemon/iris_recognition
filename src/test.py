from utils.extractandenconding import extractFeature
import cv2
import numpy as np


img = 'iris_recognition\\CASIA1\\1\\001_2_1.jpg'
eyeim = cv2.imread(img, 0)
eyelashes_threshold = 80

extractFeature(img)