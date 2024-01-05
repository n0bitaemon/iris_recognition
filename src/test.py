from utils.extractandenconding import extractFeature
from scipy.spatial import distance
import cv2
import numpy as np


img1 = 'iris_recognition\\src\\tests\\002_1_1.jpg'
img2 = 'iris_recognition\\CASIA1\\9\\009_1_1.jpg'

x1 = extractFeature(img1)
x2 = extractFeature(img2)
print(distance.hamming(x1, x2))