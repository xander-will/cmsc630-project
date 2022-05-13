
import imageio 
import json
import matplotlib.pyplot as plt
import re
from glob import glob

import operations as op
from runtime import runtime_dict

with open("config.json", "r") as f:
    config = json.load(f)

img_paths = glob("img/*.BMP")
total_msqe = 0

for img_path in img_paths:
    img_name = img_path[4:-4]  # hardcoded removal of 'img\' and '.bmp'
    print("Processing ", img_name)

    img_color = imageio.imread(img_path)
    img_grey  = imageio.imread(img_path, pilmode="L")
    print(img_grey.shape)

    ##  Salt-and-Pepper Noise
    if config["snp"]["active"]:
        out = op.SNPNoise(img_grey, config["snp"]["intensity"])
        out_path = "out/" + img_name + "snp.bmp"
        imageio.imwrite(out_path, out)

    ##  Gaussian Noise
    if config["gauss"]["active"]:
        args = [
            config["gauss"]["max"],
            config["gauss"]["min"],
            config["gauss"]["std_dev"]
        ]
        out = op.GaussNoise(img_grey, *args)
        out_path = "out/" + img_name + "gauss.bmp"
        imageio.imwrite(out_path, out)

    ##  Single Color Spectrum
    if config["color_spec"]["active"]:
        out = op.SingleColor(img_color, config["color_spec"]["color"])
        out_path = "out/" + img_name + "spec.bmp"
        imageio.imwrite(out_path, out)

    ##  Histogram
    if config["histogram"]["active"]:
        group = re.search("\D+", img_name).group(0)
        hist = op.Histogram(img_grey, group)

        if config["histogram"]["equalize"]:
            out = op.HistEqualize(img_grey, hist)
            out_path = "out/" + img_name + "equal.bmp"
            imageio.imwrite(out_path, out)

    ##  Quanitize
    if config["quanitize"]["active"]:
        out = op.Quanitize(img_grey, config["quanitize"]["delta"])
        out_path = "out/" + img_name + "quan.bmp"
        imageio.imwrite(out_path, out)

        if config["quanitize"]["msqe"]:
            total_msqe += op.MSQE(img_grey, out)

    ##  Linear
    if config["linear"]["active"]:
        out = op.Linear(img_grey, config["linear"]["filter"])
        out_path = "out/" + img_name + "lin.bmp"
        imageio.imwrite(out_path, out)

    ##  Median
    if config["median"]["active"]:
        out = op.Median(img_grey, config["median"]["filter"])
        out_path = "out/" + img_name + "med.bmp"
        imageio.imwrite(out_path, out)

    ##  Prewitt
    if config["prewitt"]["active"]:
        out = op.Prewitt(img_grey, config["prewitt"]["sigma"])
        out_path = "out/" + img_name + "pre.bmp"
        imageio.imwrite(out_path, out)

    ##  Sobel
    if config["sobel"]["active"]:
        out = op.Sobel(img_grey, config["sobel"]["sigma"])
        out_path = "out/" + img_name + "sob.bmp"
        imageio.imwrite(out_path, out)

    ##  Laplace
    if config["laplace"]["active"]:
        out = op.Laplace(img_grey, config["laplace"]["sigma"])
        out_path = "out/" + img_name + "lap.bmp"
        imageio.imwrite(out_path, out)

    ##  Dilation
    if config["dilation"]["active"]:
        out = op.Dilation(img_grey, config["dilation"]["grid_size"])
        out_path = "out/" + img_name + "dil.bmp"
        imageio.imwrite(out_path, out)

    ##  Erosion
    if config["erosion"]["active"]:
        out = op.Erosion(img_grey, config["erosion"]["grid_size"])
        out_path = "out/" + img_name + "ero.bmp"
        imageio.imwrite(out_path, out)

    ##  Thresholding Segmentation
    if config["seg_thresh"]["active"]:
        out, _ = op.ThresholdingSeg(img_grey)
        out_path = "out/" + img_name + "segth.bmp"
        imageio.imwrite(out_path, out)

    ##  Clustering Segmentation
    if config["seg_cluster"]["active"]:
        out = op.ClusteringSeg(img_color, config["seg_cluster"]["k"])
        out_path = "out/" + img_name + "segcl.bmp"
        imageio.imwrite(out_path, out)
        break

for group in op.histogram_dict:
    avg_hist = op.histogram_dict[group]
    plt.plot(avg_hist[1] / avg_hist[0])
    plt.title(f'Averaged Histogram: {group}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig('hist/' + group + '.png')
    plt.clf()

print("\n-------\n")
for oper in runtime_dict:
    print(oper)
    print(f"Overall runtime: {runtime_dict[oper]} secs")
    print(f"Average runtime: {runtime_dict[oper]/len(img_paths)} secs\n")

if config["quanitize"]["msqe"]:
    print(f"Average MSQE: {total_msqe / len(img_paths)}")