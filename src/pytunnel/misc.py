from __future__ import division, print_function

import argparse
import logging
from pathlib import Path

import yaml
from panda3d.core import loadPrcFile, Filename, getModelPath


def set_root_logger(verbose):
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    # level of verbosity
    level = logging.DEBUG if verbose else logging.WARN

    # message formating
    form = logging.Formatter(
        '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s'
    )

    # add logging to stderr
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(form)
    logger.addHandler(handler)

    return logger


def default_parser():
    parser = argparse.ArgumentParser(description="Run a virtual tunnel")
    parser.add_argument(
        'yaml_file', help="YAML file containing tunnel options"
    )
    config_path = Path(__file__).parent.parent / 'config' / 'panda3d.prc'
    parser.add_argument(
        '-c', '--panda3d-config', default=str(config_path),
        help="Panda3D configuration file (default: %(default)s)"
    )
    parser.add_argument(
        '-v', '--verbose', action="store_true",
        help="display debugging information (default: %(default)s)"
    )
    return parser


def default_main():
    args = default_parser().parse_args()
    panda3d_config = Filename.fromOsSpecific(args.panda3d_config)
    loadPrcFile(panda3d_config)
    with open(args.yaml_file, 'r') as fd:
        options = yaml.load(fd, Loader=yaml.SafeLoader)
    texture_path = options.pop('texture_path', None)
    if texture_path:
        getModelPath().appendDirectory(texture_path)
    set_root_logger(args.verbose)
    return options



def save_yaml(options):
    try:
        save_dir = Path(options['logger']['foldername'])
        save_dir.mkdir()
        file_name = 'config.yaml'
        file_path = str(save_dir / file_name)
        
        with open(file_path, 'w') as file:
            yaml.dump(options, file)
    except KeyError:
        print("foldername not specified. this session will not be saved/logged...")