import matplotlib.pyplot as plt
import numpy as np

from random import choice, randint

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