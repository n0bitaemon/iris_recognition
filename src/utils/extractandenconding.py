from utils.imgutils import segment, daugman_normalization, feature_encoding
import cv2

def extractFeature(img_filename):
    im = cv2.imread(img_filename, 0)

    #  segmentation
    segmented_iris, cirpupil, ciriris = segment(im)

    # normalization
    normalized_iris = daugman_normalization(segmented_iris, 60, 360, cirpupil[2], ciriris[2]-cirpupil[2])
    normalized_iris = cv2.cvtColor(normalized_iris, cv2.COLOR_BGR2GRAY)
    normalized_iris = cv2.equalizeHist(normalized_iris)

    #  feature encoding
    encoded_iris = feature_encoding(normalized_iris)

    #Draw image
    # cv2.imshow('initial image', im)
    # cv2.circle(im, (cirpupil[1], cirpupil[0]), cirpupil[2], (255, 0, 0), 1)
    # cv2.circle(im, (ciriris[1], ciriris[0]), ciriris[2], (255, 0, 0), 1)
    # cv2.imshow('segmented iris image', im)
    # cv2.imshow('segmented iris', segmented_iris)
    # cv2.imshow('normalized iris', normalized_iris)
    # cv2.waitKey(0)
    
    return encoded_iris
