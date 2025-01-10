#!/usr/bin/env python

"""Test if all .yaml files defining flip tunnels work (at least load).

Used as script, it will try to start all flip tunnels defined in the
repository, and report if some are not working.

At present time, it just starts the tunnel and stop it after some time to check
that loading worked.
"""

from __future__ import division, print_function

import time
import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

import yaml


def run_tunnel(tunnel_path, yaml_path, timeout):
    """Run a tunnel for some time, then give back return code and outputs."""
    # create a temporary file to replace IO module
    with yaml_path.open() as fd:
        yaml_cnt = yaml.load(fd)

    if yaml_cnt['flip_tunnel']['io_module'] == 'nidaq':
        yaml_cnt['flip_tunnel']['io_module'] = 'nidaq_stub'

    with NamedTemporaryFile(prefix='test_pytunnel_', delete=False) as tmpfd:

        if 'texture_path' in yaml_cnt:
            base_folder = os.path.relpath(
                str(yaml_path.parent), str(Path(tmpfd.name).parent)
            )
            texture_path = Path(base_folder) / yaml_cnt['texture_path']
            yaml_cnt['texture_path'] = str(texture_path)

        tmpfd.write(yaml.dump(yaml_cnt).encode('utf-8'))

    # a little bit clunky to work on python 2 and 3 :'(
    with open(os.devnull, 'w') as devnull:
        p = Popen([str(tunnel_path), tmpfd.name], stdout=devnull, stderr=PIPE)
    time.sleep(timeout)
    p.terminate()
    _, errs = p.communicate()

    # remove temporary file
    os.unlink(tmpfd.name)

    return p, errs


if __name__ == '__main__':
    # test duration for each tunnel
    timeout = 60

    # python script defining the tunnel
    tunnel_path = (
        Path(__file__).parent / '..' / 'src' / 'pytunnel' / 'flip_tunnel.py'
    )
    tunnel_path = tunnel_path.resolve()

    # find .yaml files
    folder_path = (Path(__file__).parent / '..' / 'examples').resolve()
    yaml_paths = folder_path.glob('flip_tunnel*.yaml')

    # run tunnels one by one
    good, bad = 0, 0
    for yaml_path in yaml_paths:
        print('Starting {}... '.format(yaml_path), end='')
        sys.stdout.flush()

        p, errs = run_tunnel(tunnel_path, yaml_path, timeout)

        if p.returncode > 0:
            bad += 1
            print('FAILED')
            print(errs)
        else:
            good += 1
            print('PASSED')

    # summary of success/failure
    print('{} PASSED, {} FAILED'.format(good, bad))
