from os import listdir
import cv2
# from cv2 import imread
from multiprocessing import Pool, cpu_count
from itertools import repeat
from fnmatch import filter
import numpy as np
import scipy.io as sio
import os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from scipy import signal
from scipy.ndimage import convolve
from itertools import repeat
from skimage.transform import radon
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def searchInnerBound(img):
    """
    Searching of the boundary (inner) of the iris
    """

    # integro-differential
    Y = img.shape[0] # height
    X = img.shape[1] # width
    sect = X/4 # Dividing the width into 4 equal sections
    minrad = 10
    maxrad = sect*0.8
    jump = 4 		# Precision of the search (step size)

    # Hough Space
    sz = np.array([np.floor((Y-2*sect)/jump),
                    np.floor((X-2*sect)/jump),
                    np.floor((maxrad-minrad)/jump)]).astype(int)
    # print("Y={}, X={}, sec={}, sz={}".format(Y, X, sect, sz))

    #circular integration
    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y*jump
    x = sect + x*jump
    r = minrad + r*jump
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative 
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring the image
    sm = 3
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y*jump
    inner_x = sect + x*jump
    inner_r = minrad + (r-1)*jump

    # Integro-Differential 
    integrationprecision = 0.1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump*2),
                          np.arange(jump*2),
                          np.arange(jump*2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative 
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring the image
    sm = 3 	
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r

def searchOuterBound(img, inner_y, inner_x, inner_r):
    """
    Searching fo the boundary (outer) of the iris 
    """
    maxdispl = np.round(inner_r*0.15).astype(int)

    minrad = np.round(inner_r/0.8).astype(int)
    maxrad = np.round(inner_r/0.3).astype(int)

    # Integration region and avoiding eyelids
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    #circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring
    sm = 7 	# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y, x ,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r

def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    """
       Contour/circular integral using discrete rieman
    """
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)
    
    # adapt x and y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    hs = imagen[y, x]

    hs = np.sum(hs, axis=0)
    return hs.astype(float)

def findTopEyelid(imsz, imageiris, irl, icl, rowp, rp, ret_top=None):
    """
    Find and mask for the top eyelid region.
    """
    topeyelid = imageiris[0: rowp - irl - rp, :]
    lines = findline(topeyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, topeyelid.shape)
        yl = np.round(yl + irl - 1).astype(int)
        xl = np.round(xl + icl - 1).astype(int)

        yla = np.max(yl)
        y2 = np.arange(yla)

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        print('grid = {}, mask size = {}'.format(grid, mask.shape))
        mask[grid] = np.nan

    if ret_top is not None:
        ret_top[0] = mask
    return mask

def findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl, ret_bot=None):
    """
    Find and mask for the bottom eyelid region.
    """
    bottomeyelid = imageiris[rowp - irl + rp - 1 : imageiris.shape[0], :]
    lines = findline(bottomeyelid)
    mask = np.zeros(imsz, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, bottomeyelid.shape)
        yl = np.round(yl + rowp + rp - 3).astype(int)
        xl = np.round(xl + icl - 2).astype(int)
        yla = np.min(yl)
        y2 = np.arange(yla-1, imsz[0])

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        mask[grid] = np.nan

    if ret_bot is not None:
        ret_bot[0] = mask
    return mask

def findline(img):
    """
    Find lines in the image using linear hough transformation and 
    canny detection
    """
    I2, orient = canny(img, 2, 0, 1)
    I3 = adjgamma(I2, 1.9)
    I4 = nonmaxsup(I3, orient, 1.5)
    edgeimage = hysthresh(I4, 0.2, 0.15)

    # Radon transformation
    theta = np.arange(180)
    R = radon(edgeimage, theta, circle=False)
    sz = R.shape[0] // 2
    xp = np.arange(-sz, sz+1, 1)

    maxv = np.max(R)
    if maxv > 25:
        i = np.where(R.ravel() == maxv)
        i = i[0]
    else:
        return np.array([])

    R_vect = R.ravel()
    ind = np.argsort(-R_vect[i])
    u = i.shape[0]
    k = i[ind[0: u]]
    y, x = np.unravel_index(k, R.shape)
    t = -theta[x] * np.pi / 180
    r = xp[y]

    lines = np.vstack([np.cos(t), np.sin(t), -r]).transpose()
    cx = img.shape[1] / 2 - 1
    cy = img.shape[0] / 2 - 1
    lines[:, 2] = lines[:, 2] - lines[:, 0]*cx - lines[:, 1]*cy
    return lines

def linecoords(lines, imsize):
    """
    Find x-, y- coordinates of positions along in a line.
    """
    xd = np.arange(imsize[1])
    yd = (-lines[0, 2] - lines[0, 0] * xd) / lines[0, 1]

    coords = np.where(yd >= imsize[0])
    coords = coords[0]
    yd[coords] = imsize[0]-1
    coords = np.where(yd < 0)
    coords = coords[0]
    yd[coords] = 0

    x = xd
    y = yd
    return x, y

def canny(im, sigma, vert, horz):
    """
    Canny edge detection.
    """
    def fspecial_gaussian(shape=(3, 3), sig=1):
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        f = np.exp(-(x * x + y * y) / (2 * sig * sig))
        f[f < np.finfo(f.dtype).eps * f.max()] = 0
        sum_f = f.sum()
        if sum_f != 0:
            f /= sum_f
        return f

    hsize = [6 * sigma + 1, 6 * sigma + 1] 
    gaussian = fspecial_gaussian(hsize, sigma)
    im = convolve(im, gaussian, mode='constant')  
    rows, cols = im.shape

    h = np.concatenate([im[:, 1:cols], np.zeros([rows, 1])], axis=1) - \
        np.concatenate([np.zeros([rows, 1]), im[:, 0: cols - 1]], axis=1)

    v = np.concatenate([im[1: rows, :], np.zeros([1, cols])], axis=0) - \
        np.concatenate([np.zeros([1, cols]), im[0: rows - 1, :]], axis=0)

    d11 = np.concatenate([im[1:rows, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d11 = np.concatenate([d11, np.zeros([1, cols])], axis=0)
    d12 = np.concatenate([np.zeros([rows-1, 1]), im[0:rows - 1, 0:cols - 1]], axis = 1)
    d12 = np.concatenate([np.zeros([1, cols]), d12], axis=0)
    d1 = d11 - d12

    d21 = np.concatenate([im[0:rows - 1, 1:cols], np.zeros([rows - 1, 1])], axis = 1)
    d21 = np.concatenate([np.zeros([1, cols]), d21], axis=0)
    d22 = np.concatenate([np.zeros([rows - 1, 1]), im[1:rows, 0:cols - 1]], axis = 1)
    d22 = np.concatenate([d22, np.zeros([1, cols])], axis=0)
    d2 = d21 - d22

    X = (h + (d1 + d2) / 2) * vert
    Y = (v + (d1 - d2) / 2) * horz

    gradient = np.sqrt(X * X + Y * Y)  

    orient = np.arctan2(-Y, X)
    neg = orient < 0 
    orient = orient * ~neg + (orient + np.pi) * neg
    orient = orient * 180 / np.pi

    return gradient, orient

def adjgamma(im, g):
    """
    Adjust image gamma.
    """
    newim = im
    newim = newim - np.min(newim)
    newim = newim / np.max(newim)
    newim = newim ** (1 / g) 
    return newim

def nonmaxsup(in_img, orient, radius):
    """
    Perform non-maxima suppression on an image using an orientation image
    """
    rows, cols = in_img.shape
    im_out = np.zeros([rows, cols])
    iradius = np.ceil(radius).astype(int)

    # precalculatihg x and y offsets to relatives to the center piuxel
    angle = np.arange(181) * np.pi / 180 
    xoff = radius * np.cos(angle)
    yoff = radius * np.sin(angle)
    hfrac = xoff - np.floor(xoff)
    vfrac = yoff - np.floor(yoff)
    orient = np.fix(orient)

    # interpolating grey values of the center pixel for the nom maximal suppression
    col, row = np.meshgrid(np.arange(iradius, cols - iradius),
                           np.arange(iradius, rows - iradius))

    ori = orient[row, col].astype(int)
    x = col + xoff[ori]
    y = row - yoff[ori]
    # pixel locations that surround location x,y
    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)
    # integer pixel locations
    bl = in_img[cy, fx]  # bottom left
    br = in_img[cy, cx]  # bottom right
    tl = in_img[fy, fx]  # top left
    tr = in_img[fy, cx]  # top right
    # Bi-linear interpolation for x,y values
    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # same thing but for the other side
    map_candidate_region = in_img[row, col] > v1
    x = col - xoff[ori]
    y = row + yoff[ori]
    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)
    tl = in_img[fy, fx]
    tr = in_img[fy, cx]
    bl = in_img[cy, fx]
    br = in_img[cy, cx]
    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # max local
    map_active = in_img[row, col] > v2
    map_active = map_active * map_candidate_region
    im_out[row, col] = in_img[row, col] * map_active

    return im_out

def hysthresh(im, T1, T2):
    """
    Hysteresis thresholding.
    """
    rows, cols = im.shape
    rc = rows * cols
    rcmr = rc - rows
    rp1 = rows + 1

    bw = im.ravel() # column vector 
    pix = np.where(bw > T1)  # pixels with value > T1
    pix = pix[0]
    npix = pix.size         # pixels with value > T1

    # stack array
    stack = np.zeros(rows * cols)
    stack[0:npix] = pix         # add edge points on the stack
    stp = npix  
    for k in range(npix):
        bw[pix[k]] = -1        

    O = np.array([-1, 1, -rows - 1, -rows, -rows + 1, rows - 1, rows, rows + 1])

    while stp != 0:  # While the stack is != empty
        v = int(stack[stp-1])  
        stp -= 1

        if rp1 < v < rcmr:  # prevent illegal indices
            index = O + v  # indices of points around this pixel.
            for l in range(8):
                ind = index[l]
                if bw[ind] > T2:  # value > T2,
                    stp += 1  # add index onto the stack.
                    stack[stp-1] = ind
                    bw[ind] = -1 

    bw = (bw == -1)  # zero out that was not an edge
    bw = np.reshape(bw, [rows, cols])  # Reshaping the image
    
    return bw


def circlecoords(c, r, imgsize, nsides=600):
    """
    Find the coordinates of a circle based on its centre and radius.
    """
    a = np.linspace(0, 2*np.pi, 2*nsides+1)
    xd = np.round(r * np.cos(a) + c[0])
    yd = np.round(r * np.sin(a) + c[1])

    # remove value bigger than the image
    xd2 = xd
    coords = np.where(xd >= imgsize[1])
    xd2[coords[0]] = imgsize[1] - 1
    coords = np.where(xd < 0)
    xd2[coords[0]] = 0

    yd2 = yd
    coords = np.where(yd >= imgsize[0])
    yd2[coords[0]] = imgsize[0] - 1
    coords = np.where(yd < 0)
    yd2[coords[0]] = 0

    x = np.round(xd2).astype(int)
    y = np.round(yd2).astype(int)
    return x, y

def segment(eyeim, eyelashes_thres=80, use_multiprocess=True):
    rowp, colp, rp = searchInnerBound(eyeim)
    row, col, r = searchOuterBound(eyeim, rowp, colp, rp)

    # pupil and iris boundaries
    rowp = np.round(rowp).astype(int)
    colp = np.round(colp).astype(int)
    rp = np.round(rp).astype(int)
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)
    r = np.round(r).astype(int)
    cirpupil = [rowp, colp, rp]
    ciriris = [row, col, r]

    # top and bottom eyelid
    imsz = eyeim.shape
    irl = np.round(row - r).astype(int)
    iru = np.round(row + r).astype(int)
    icl = np.round(col - r).astype(int)
    icu = np.round(col + r).astype(int)
    if irl < 0:
        irl = 0
    if icl < 0:
        icl = 0
    if iru >= imsz[0]:
        iru = imsz[0] - 1
    if icu >= imsz[1]:
        icu = imsz[1] - 1
    imageiris = eyeim[irl: iru + 1, icl: icu + 1]

    # Draw
    # testim = eyeim
    # window_name = 'Iris with bounds'
    # testim = cv2.circle(testim, (cirpupil[1], cirpupil[0]), cirpupil[2], (0, 0, 125), 1)
    # testim = cv2.circle(testim, (ciriris[1], ciriris[0]), ciriris[2], (0, 125, 0), 1)
    # cv2.imshow(window_name, testim)
    # cv2.imshow('Iris bounded', imageiris)
    # cv2.waitKey(0)

    mask_top = findTopEyelid(imsz, imageiris, irl, icl, rowp, rp)
    mask_bot = findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl)

    # noise region we mark by NaN value
    imwithnoise = eyeim.astype(float)
    imwithnoise = imwithnoise + mask_top + mask_bot

    # For CASIA dataset, we need to eliminate eyelashes by threshold
    ref = eyeim < eyelashes_thres
    coords = np.where(ref == 1)
    imwithnoise[coords] = np.nan

    return ciriris, cirpupil, imwithnoise


def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil,
              radpixels, angulardiv):
    """
    Turn a circular region into a rectangular block of NxN dimensions
    """
    radiuspixels = radpixels + 2
    angledivisions = angulardiv-1

    r = np.arange(radiuspixels)
    theta = np.linspace(0, 2*np.pi, angledivisions+1)

    # displacement of pupil and iris center
    ox = x_pupil - x_iris
    oy = y_pupil - y_iris

    if ox <= 0:
        sgn = -1
    elif ox > 0:
        sgn = 1

    if ox == 0 and oy > 0:
        sgn = 1

    a = np.ones(angledivisions+1) * (ox**2 + oy**2)

    if ox == 0:
        phi = np.pi/2
    else:
        phi = np.arctan(oy/ox)

    b = sgn * np.cos(np.pi - phi - theta)

    r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - r_iris**2))
    r = np.array([r - r_pupil])

    rmat = np.dot(np.ones([radiuspixels, 1]), r)

    rmat = rmat * np.dot(np.ones([angledivisions + 1, 1]), np.array([np.linspace(0, 1, radiuspixels)])).transpose()
    rmat = rmat + r_pupil

    # exclude teh boundarys of the pupil iris border 
    rmat = rmat[1:radiuspixels-1, :]

    # cartesian location of each point around the iris
    xcosmat = np.dot(np.ones([radiuspixels-2, 1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([radiuspixels-2, 1]), np.array([np.sin(theta)]))
    xo = rmat * xcosmat
    yo = rmat * xsinmat
    xo = x_pupil + xo
    xo = np.round(xo).astype(int)
    coords = np.where(xo >= image.shape[1])
    xo[coords] = image.shape[1] - 1
    coords = np.where(xo < 0)
    xo[coords] = 0
    
    yo = y_pupil - yo
    yo = np.round(yo).astype(int)
    coords = np.where(yo >= image.shape[0])
    yo[coords] = image.shape[0] - 1
    coords = np.where(yo < 0)
    yo[coords] = 0

    polar_array = image[yo, xo]
    polar_array = polar_array / 255

    # noise array with location of NaNs in polar_array
    polar_noise = np.zeros(polar_array.shape)
    coords = np.where(np.isnan(polar_array))
    polar_noise[coords] = 1

    # Get rid of outling points
    image[yo, xo] = 255

    # Get pixel coords for iris
    x, y = circlecoords([x_iris, y_iris], r_iris, image.shape)
    image[y, x] = 255

    xp, yp = circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
    image[yp, xp] = 255

    # Replace NaNs before performing feature encoding
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg

    return polar_array, polar_noise.astype(bool)

def encode_iris(arr_polar, arr_noise, minw_length, mult, sigma_f):
    """
    Generate iris template and noise mask from the normalised iris region.
    """
    # convolve with gabor filters
    filterb = gaborconvolve_f(arr_polar, minw_length, mult, sigma_f)
    l = arr_polar.shape[1]
    template = np.zeros([arr_polar.shape[0], 2 * l])
    h = np.arange(arr_polar.shape[0])

    # making the iris template
    mask_noise = np.zeros(template.shape)
    filt = filterb[:, :]

    # quantization and check to se if the phase data is useful
    H1 = np.real(filt) > 0
    H2 = np.imag(filt) > 0

    H3 = np.abs(filt) < 0.0001
    for i in range(l):
        ja = 2 * i

        # biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]
        # noise mask_noise
        mask_noise[:, ja] = arr_noise[:, i] | H3[:, i]
        mask_noise[:, ja + 1] = arr_noise[:, i] | H3[:, i]

    return template, mask_noise

def gaborconvolve_f(img, minw_length, mult, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component 
    fo = 1 / wavelength
    logGabor_f[0: int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) /
                                    (2 * np.log(sigma_f)**2))
    logGabor_f[0] = 0

    # convolution for each row
    for r in range(rows):
        signal = img[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)
    
    return filterb

def extractFeature(img_filename, eyelashes_threshold=80, multiprocess=True):
    """
    Extract features from an iris image
    """
    # parameters
    eyelashes_threshold = 80
    radial_resolution = 20
    angular_resolution = 240
    minw_length = 18
    mult = 1
    sigma_f = 0.5

    #  segmentation
    im = cv2.imread(img_filename, 0)
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_threshold,
                                    multiprocess)

    # normalization
    arr_polar, arr_noise = normalize(imwithnoise, ciriris[1],  ciriris[0], ciriris[2],
                                         cirpupil[1], cirpupil[0], cirpupil[2],
                                         radial_resolution, angular_resolution)

    #  feature encoding
    template, mask_noise = encode_iris(arr_polar, arr_noise, minw_length, mult,
    sigma_f)
    

    return template, mask_noise, img_filename

def matchingTemplate(template_extr, mask_extr, template_dir, threshold=0.38):
    """
    Matching the template of the image with the ones in the database
    """
    # n# of accounts in the database
    n_files = len(filter(listdir(template_dir), '*.mat'))
    if n_files == 0:
        return -1

    # use every cores to calculate Hamming distances
    args = zip(
        sorted(listdir(template_dir)),
        repeat(template_extr),
        repeat(mask_extr),
        repeat(template_dir),
    )
    with Pool(processes=cpu_count()) as pools:
        result_list = pools.starmap(matchingPool, args)

    filenames = [result_list[i][0] for i in range(len(result_list))]
    hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

    # Removing NaN elements
    ind_valid = np.where(hm_dists > 0)[0]
    hm_dists = hm_dists[ind_valid]
    filenames = [filenames[idx] for idx in ind_valid]

    ind_thres = np.where(hm_dists <= threshold)[0]
    if len(ind_thres) == 0:
        return 0
    else:
        hm_dists = hm_dists[ind_thres]
        filenames = [filenames[idx] for idx in ind_thres]
        ind_sort = np.argsort(hm_dists)
        return [filenames[idx] for idx in ind_sort]

def matchingPool(file_temp_name, template_extr, mask_extr, template_dir):
    """
    Perform matching session within a Pool of parallel computation
    """
    data_template = sio.loadmat('%s%s' % (template_dir, file_temp_name))
    template = data_template['template']
    mask = data_template['mask']

    # the Hamming distance
    hm_dist = HammingDistance(template_extr, mask_extr, template, mask)
    return (file_temp_name, hm_dist)


def HammingDistance(template1, mask1, template2, mask2):
    """
    Calculate the Hamming distance between two iris templates.
    """
    hd = np.nan

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-8, 9):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1

    return hd


def shiftbits_ham(template, noshifts):
    """
    Shift the bit-wise iris patterns.
    """
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew

# extractFeature('/home/kali/Documents/IrisRecognition/CASIA1/95/095_1_2.jpg', 0)
if __name__ == '__main__':
    dataset_dir = 'CASIA1 copy/*'
    template_dir = 'templates/CASIA1/'

    if not os.path.exists(template_dir):
        print("makedirs", template_dir)
        os.makedirs(template_dir)

    files = glob(os.path.join(dataset_dir, "*_1_*.jpg"))
    n_files = len(files)
    print("N# of files which we are extracting features", n_files)

    for idx, file in enumerate(files):
        print("#{}. Process file {}... => ".format(idx, file), end='')
        try:
            template, mask, _ = extractFeature(file, multiprocess=False)
            basename = os.path.basename(file)
            out_file = os.path.join(template_dir, "%s.mat" % (basename))
            savemat(out_file, mdict={'template': template, 'mask': mask})
            print('Success')
        except:
            print('Error')
            pass