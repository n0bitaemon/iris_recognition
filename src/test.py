from utils.imgutils import segment, daugman_normalizaiton
import cv2
import numpy as np


img = 'iris_recognition\\CASIA1\\44\\044_1_1.jpg'
eyeim = cv2.imread(img, 0)
eyelashes_threshold = 80

# rowp, colp, rp = segment(eyeim)
# print('rowp = {}, colp = {}, rp = {}'.format(rowp, colp, rp))

irisimage, cirpupil, ciriris = segment(eyeim)
image_nor = daugman_normalizaiton(irisimage, 60, 360, cirpupil[2], ciriris[2]-cirpupil[2])
image_nor = cv2.cvtColor(image_nor, cv2.COLOR_BGR2GRAY)
image_nor = cv2.equalizeHist(image_nor)
cv2.imshow('aaa', image_nor)
cv2.imshow('iris', irisimage)
cv2.waitKey(0)