
import imageio 
import json
import matplotlib as mpl
import re
from glob import glob

import operations as op
from runtime import runtime_dict

with open("config.json", "r") as f:
    config = json.load(f)

img_paths = glob("img/*.BMP")

for img_path in img_paths:
    img_name = img_path[4:-4]  # hardcoded removal of 'img\' and '.bmp'

    img_color = imageio.imread(img_path)
    img_grey  = imageio.imread(img_path, pilmode="L")

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

    ##  Linear
    if config["linear"]["active"]:
        out = op.Linear(img_grey, config["linear"]["filter"])
        out_path = "out/" + img_name + "lin.bmp"
        imageio.imwrite(out_path, out)

##  TODO: create averaged histogram pics

for oper in runtime_dict:
    print(oper)
    print(f"Overall runtime: {runtime_dict[oper]} secs")
    print(f"Average runtime: {runtime_dict[oper]/len(img_paths)} secs")
