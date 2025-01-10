#!/usr/bin/env python

from __future__ import division, print_function

import sys
import logging
from math import log10, floor
from pathlib import Path

import yaml
from panda3d.core import loadPrcFile, Filename

from misc import default_parser, set_root_logger
from flip_tunnel import make_flip_tunnel


if __name__ == "__main__":
    # script options
    parser = default_parser()
    parser.add_argument('video_folder', help='folder where to save frames')
    parser.add_argument('-d', '--duration', default=150, type=int,
                        help='movie duration (default: %(default)s)')
    parser.add_argument('-f', '--fps', default=30, type=int,
                        help='frames per second (default: %(default)s)')
    args = parser.parse_args()

    # load configuration files
    panda3d_config = Filename.fromOsSpecific(args.panda3d_config)
    loadPrcFile(panda3d_config)
    with open(args.yaml_file, 'r') as fd:
        options = yaml.load(fd)

    # create movie folder, and complain if already exist (avoid overriding)
    folder_path = Path(args.video_folder)
    if folder_path.exists():
        print("Movie folder '{}' already exists! Please choose another folder."
              .format(folder_path))
        sys.exit(1)

    folder_path.mkdir(parents=True)

    # save logs to a file
    logger = set_root_logger(args.verbose)

    form = logging.Formatter(
        '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s'
    )

    handler = logging.FileHandler(str(folder_path / 'log.txt'), mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(form)
    logger.addHandler(handler)

    # start tunnel with additional task to record frames
    tunnel = make_flip_tunnel(
        options['base_tunnel'], options['card'], options['flip_tunnel'],
        options['walls_sequence'], options['inputs'], options['outputs'],
        test_mode=False
    )

    prefix = str(folder_path / 'frame')
    sd = floor(log10(args.fps * args.duration) + 1)
    tunnel.tunnel.movie(prefix, duration=args.duration, fps=args.fps, sd=sd)
    tunnel.tunnel.taskMgr.doMethodLater(
        args.duration, sys.exit, 'exit_task', extraArgs=[], sort=50
    )

    tunnel.run()
