#!/usr/bin/env python

import sys

import numpy as np
from scipy.misc import imread
from scipy.io import savemat
from pathlib import Path


if __name__ == '__main__':
    # define input folder and output file
    if len(sys.argv) != 3:
        print('usage: extract_grey.py <video_dir> <result_file.mat>')
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    # extract position information from log file
    with (folder_path / 'log.txt').open('r') as fd:
        txt = fd.read()

    lines = [line for line in txt.split('\n') if 'position' in line]
    position = [float(l.split('position: ')[-1].split(',')[0]) for l in lines]

    # extract grey intensity from images
    average = []
    luminosity = []

    for p in folder_path.glob('*.png'):
        print(p)
        img = imread(str(p))
        img = img[:, :img.shape[1]//2, :3]

        average.append(img.mean(-1).mean())
        luminosity.append(img.dot([0.2125, 0.7154, 0.0721]).mean())

    average = np.array(average)
    luminosity = np.array(luminosity)

    # save results
    savemat(
        str(result_path),
        {'position': position, 'average': average, 'luminosity': luminosity}
    )
