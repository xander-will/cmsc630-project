import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.ndimage.filters import convolve


from random import choice, gauss, randint

from runtime import timer

@timer
def SNPNoise(img, intensity):
    out = np.ndarray.copy(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if randint(1, 101) <= intensity:
                out[i][j] = choice([0, 255])

    return out

@timer
def GaussNoise(img, max_int, min_int, std_dev):
    noise = np.random.normal(0, std_dev, img.shape)
    noise += -np.amin(noise) + min_int
    noise *= max_int / np.amax(noise)
    
    return img + np.rint(noise)

@timer
def SingleColor(img, color):
    out = np.ndarray.copy(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if color == "red":
                out[i][j] = [out[i][j][0], 0, 0]
            elif color == "green":
                out[i][j] = [0, out[i][j][1], 0]
            elif color == "blue":
                out[i][j] = [0, 0, out[i][j][2]]

    return out

histogram_dict = {}

def _my_hist(img):
    hist = np.zeros(256)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1

    return hist

@timer
def Histogram(img, group):
    hist = _my_hist(img)

    try:
        histogram_dict[group][0] += 1
        histogram_dict[group][1] += hist
    except KeyError:
        histogram_dict[group] = [1, hist]
        
    return hist

@timer
def HistEqualize(img, hist):
    cdf = np.cumsum(hist)
    cdf_min = np.min(cdf[np.nonzero(cdf)])
    denom = 255 / (img.shape[0]*img.shape[1] - cdf_min)  # taken from wikipedia

    h = np.copy(cdf)
    for i in range(len(cdf)):
        h[i] = np.rint((cdf[i] - cdf_min) * denom)
    
    out = np.copy(img)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = h[out[i][j]]
    
    return out

@timer
def Quanitize(img, delta):
    return delta*np.rint(img / delta) + delta//2

def MSQE(img, q_img):
    sum = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum += (img[i,j] - q_img[i,j])**2 / 256

    return sum / (img.shape[0]*img.shape[1])

@timer
def Linear(img, filter):
    filter = np.array(filter)
    f_size = filter.shape[0] * filter.shape[1]
    v, h = (filter.shape[0] - 1) // 2, (filter.shape[1] - 1) // 2

    out = np.copy(img)
    for i in range(v, img.shape[0]-v):
        for j in range(h, img.shape[1]-h):
            window = img[i-v:i+v+1, j-h:j+h+1] * filter
            out[i][j] = np.mean(window)

    return out

def _weighted_median(window, filter):
    full = np.zeros(np.sum(filter))
    window, filter = np.ndarray.flatten(window), np.ndarray.flatten(filter)

    j = 0
    for i in range(len(window)):
        for _ in range(filter[i]):
            full[j] = window[i]
            j += 1

    return np.median(full)

@timer
def Median(img, filter):
    filter = np.array(filter)
    f_size = filter.shape[0] * filter.shape[1]
    v, h = (filter.shape[0] - 1) // 2, (filter.shape[1] - 1) // 2

    out = np.copy(img)
    for i in range(v, img.shape[0]-v):
        for j in range(h, img.shape[1]-h):
            window = img[i-v:i+v+1, j-h:j+h+1]
            out[i][j] = _weighted_median(window, filter)

    return out

def _convolve(img, filter):
    v, h = (filter.shape[0] - 1) // 2, (filter.shape[1] - 1) // 2

    out = np.copy(img)
    for i in range(v, img.shape[0]-v):
        for j in range(h, img.shape[1]-h):
            window = img[i-v:i+v+1, j-h:j+h+1] * filter
            out[i][j] = np.sum(window)

    return out

def _pixel_axis_map(angle):
    if angle < 0:
        angle += 2*math.pi
    
    if angle > (7 * math.pi / 8):
        return ((0,1), (2,1))
    elif angle > (5 * math.pi / 8):
        return ((0,0), (2,2))
    elif angle > (3 * math.pi / 8):
        return ((1,0), (1,2))
    elif angle > (1 * math.pi / 8):
        return ((2,0), (0,2))
    else:
        return ((0,1), (2,1))

@timer
def Prewitt(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_xderiv = convolve(img, sobel_x)
    img_yderiv = convolve(img, sobel_y)

    return np.rint((img_xderiv + img_yderiv) / 2)

def _gauss_filt(img, sigma):
    sig2 = 2 * (sigma ** 2)
    gauss_filt = [[math.exp(-((i**2 + j**2) / sig2)  / (math.pi * sig2))
                    for j in range(5)] 
                    for i in range(5)]
    gauss_filt = np.array(gauss_filt)
    gauss_filt /= math.pi * sig2
    return(convolve(img, gauss_filt))

@timer
def Prewitt(img, sigma):
    img = _gauss_filt(img, sigma)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_xderiv = convolve(img, prewitt_x)
    img_yderiv = convolve(img, prewitt_y)

    return np.rint(np.hypot(img_xderiv, img_yderiv))

@timer
def Sobel(img, sigma):
    img = _gauss_filt(img, sigma)    

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_xderiv = convolve(img, sobel_x)
    img_yderiv = convolve(img, sobel_y)

    return np.rint(np.hypot(img_xderiv, img_yderiv))

@timer
def Laplace(img, sigma):
    img = _gauss_filt(img, sigma)

    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return np.rint(convolve(img, laplace))

@timer
def Dilation(img, g):
    out = np.copy(img)
    for i in range(g, img.shape[0]-g):
        for j in range(g, img.shape[1]-g):
            out[i][j] = np.min(img[i-g:i+g+1, j-g:j+g+1])

    return out

@timer
def Erosion(img, g):
    out = np.copy(img)
    for i in range(g, img.shape[0]-g):
        for j in range(g, img.shape[1]-g):
            out[i][j] = np.max(img[i-g:i+g+1, j-g:j+g+1])

    return out

@timer
def ThresholdingSeg(img):
    P = _my_hist(img) / (img.shape[0] * img.shape[1])
    p_o = np.cumsum(P)
    p_o = np.where(p_o != 0, p_o, 0.00001)  # removes divide by zero errors
    p_b = 1 - p_o
    p_b = np.where(p_b != 0, p_b, 0.00001)

    iP = np.cumsum(np.arange(256) * P)
    mu_o = iP / p_o
    mu_b = (iP[255] - iP) / p_b

    imu_oP = np.cumsum((np.arange(256) - mu_o) ** 2 * P)
    imu_bP = np.cumsum((np.arange(256) - mu_b) ** 2 * P)
    sig_o = imu_oP / p_o
    sig_b = (imu_bP[255] - imu_bP) / p_b

    sig_w = sig_o * p_o + sig_b * p_b
    min_T = np.argmin(sig_w)

    return np.where(img < min_T, 0, 255), min_T

def _find_closest_mean(elem, means):
    results = [np.linalg.norm(elem - mean) for mean in means]
    return np.argmin(np.array(results))

@timer
def ClusteringSeg(img, k):
    new_means = np.array([np.array((randint(0, 255), randint(0, 255), randint(0, 255))) for _ in range(k)])
    closest = np.zeros(img.shape[0:2])
    means = np.array(([np.array([-1, -1, -1]) for _ in range(k)]))

    while np.sum(np.abs(means - new_means)) != 0:
        print("start of k iteration loop")
        np.copyto(means, new_means)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                closest[i, j] = _find_closest_mean(img[i, j], means)

        for i in range(k):
            cluster = img[np.where(closest == i)]
            new_means[i] = np.rint(np.average(cluster, axis=0))

    out = np.zeros(img.shape)
    for i, mean in enumerate(means):
        out[np.where(closest)] = mean

    return out

@timer
def Canny(img, sigma):
    sig2 = 2 * (sigma ** 2)
    gauss_filt = [[math.exp(-(((i-3)**2 + (j-3)**2) / sig2) * (1.0 / (math.pi * sig2)))
                    for j in range(5)] 
                    for i in range(5)]
    gauss_filt = np.array(gauss_filt)
    gauss_filt /= math.pi * sig2
    img_filt = convolve(img, gauss_filt)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_xderiv = convolve(img_filt, sobel_x)
    img_yderiv = convolve(img_filt, sobel_y)

    img_grad = np.hypot(img_xderiv, img_yderiv)
    img_grad = np.rint(255 * (img_grad / np.max(img_grad)))
    img_suppr = np.zeros(img_grad.shape)
    for i in range(1, img_grad.shape[0]-1):
        for j in range(1, img_grad.shape[1]-1):
            angle = np.arctan2(img_xderiv[i, j], img_yderiv[i, j])
            pix1, pix2 = _pixel_axis_map(angle)

            neighbors = img[i-1:i+2, j-1:j+2]
            if max(neighbors[pix1], img_grad[i, j], neighbors[pix2]) == img_grad[i, j]:
                img_suppr[i, j] = img_grad[i, j]
    
    thresh_low, thresh_high = 50, 100
    img_thresh = np.zeros(img_suppr.shape)
    for i in range(img_suppr.shape[0]):
        for j in range(img_suppr.shape[1]):
            if img_suppr[i, j] >= thresh_high:
                img_thresh[i, j] = 255
            elif img_suppr[i, j] >= thresh_low:
                img_thresh[i, j] = 100

    for i in range(1, img_thresh.shape[0]-1):
        for j in range(1, img_thresh.shape[1]-1):
            if img_thresh[i, j] == 100:
                if ((img_thresh[i+1, j-1] == 255) or (img_thresh[i+1, j] == 255) or (img_thresh[i+1, j+1] == 255)
                            or (img_thresh[i, j-1] == 255) or (img_thresh[i, j+1] == 255)
                            or (img_thresh[i-1, j-1] == 255) or (img_thresh[i-1, j] == 255) or (img_thresh[i-1, j+1] == 255)):
                    img[i, j] = 255
            #if img_thresh[i, j] == 100 and (255 in img_thresh[i-1:i+2, j-1:j+2]):
            #    img_thresh[i, j] = 255 ## I don't know why this doesn't work and it's driving me crazy
            else:
                img_thresh[i, j] = 0

    return img_thresh
