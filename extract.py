import csv
import imageio 
import json
import matplotlib.pyplot as plt
import numpy as np
import re
from glob import glob

import operations as op

def ExtractName(path):
    path = path[4:-4]
    group = re.search("\D+", path).group(0)
    id = path[len(group):]

    return group, int(id)

img_paths = glob("img/*.BMP")
rows = []

for i, img_path in enumerate(img_paths):
    name, id = ExtractName(img_path)  # hardcoded removal of 'img\' and '.bmp'
    print("Processing ", name + str(id))

    img_grey  = imageio.imread(img_path, pilmode="L")
    img_seg, thresh = op.ThresholdingSeg(img_grey)
    img_ero = op.Erosion(img_seg, 3)

    area = np.count_nonzero(img_seg == 0)
    perimeter = area - np.count_nonzero(img_ero == 0)
    std_dev = np.std(np.where(img_grey < thresh))
    mean_cell = np.mean(np.where(img_grey < thresh))
    mean_bkgd = np.mean(np.where(img_grey > thresh))

    rows.append([i, area, perimeter, std_dev, mean_cell, mean_bkgd, name])



with open('dataset.csv', mode='w') as dataset:
    writer = csv.writer(dataset)
    writer.writerows(rows)

    