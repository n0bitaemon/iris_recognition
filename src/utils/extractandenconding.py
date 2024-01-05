from os import listdir
from utils.imgutils import segment, daugman_normalization, feature_extraction
import cv2

def extractFeature(img_filename):
    im = cv2.imread(img_filename, 0)

    #  segmentation
    segmented_iris, cirpupil, ciriris = segment(im)

    # normalization
    normalized_iris = daugman_normalization(segmented_iris, 60, 360, cirpupil[2], ciriris[2]-cirpupil[2])

    #  feature encoding
    filtered_iris = feature_extraction(normalized_iris)

    #Draw image
    cv2.imshow('segmented iris', segmented_iris)
    cv2.imshow('normalized iris', normalized_iris)
    cv2.imshow('extracted iris', filtered_iris)
    cv2.waitKey(0)
    
    return