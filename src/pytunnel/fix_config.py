#!/usr/bin/env python

import sys
from collections import OrderedDict

import yaml


# NOQA taken from https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


yaml.add_representer(OrderedDict, represent_ordereddict)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: fix_config.py <input_config.yaml> <output_config.yaml>')
        sys.exit(1)

    # load old format configuration file
    with open(sys.argv[1], 'r') as fd:
        options = yaml.load(fd)

    # transform for new format
    options_new = OrderedDict()
    options_new['base_tunnel'] = OrderedDict()
    if 'bg_color' in options:
        options_new['base_tunnel']['bg_color'] = options.pop('bg_color')
    if 'speed_gain' in options:
        options_new['base_tunnel']['speed_gain'] = options.pop('speed_gain')
    if 'eye_fov' in options:
        options_new['base_tunnel']['eye_fov'] = options.pop('eye_fov')

    options_new['card'] = options.pop('card')
    options_new['flip_tunnel'] = OrderedDict()
    options_new['flip_tunnel']['sleep_time'] = options.pop('sleep_time')
    options_new['flip_tunnel']['stimulus_onset'] = \
        options.pop('stimulus_onset')
    options_new['flip_tunnel']['neutral_texture'] = \
        options.pop('neutral_texture')
    options_new['flip_tunnel']['io_module'] = 'nidaq'

    options_new['walls_sequence'] = options.pop('walls_sequence')
    options_new['inputs'] = OrderedDict()
    options_new['inputs']['speed'] = options.pop('encoder_chan')
    if 'speed_threshold' in options:
        options_new['inputs']['speed']['threshold'] = \
            options.pop('speed_threshold')
    if 'lick_chan' in options:
        options_new['inputs']['lick'] = options.pop('lick_chan')
        options_new['inputs']['lick']['threshold'] = \
            options.pop('lick_threshold')
    options_new['outputs'] = OrderedDict()
    options_new['outputs']['speed'] = options.pop('speed_chan')
    options_new['outputs']['position'] = options.pop('position_chan')
    options_new['outputs']['stim_id'] = options.pop('stim_id_chan')

    # display non-converted options
    print('Remaining options:', options)

    # save new configuration file
    options_yaml = yaml.dump(options_new)

    options_lines = options_yaml.split('\n')
    for i, line in enumerate(options_lines):
        if line.startswith('  '):
            options_lines[i] = line.replace('  ', '    ')
    for i, line in enumerate(options_lines):
        options_lines[i] = line.replace('- ', '  - ')

    with open(sys.argv[2], 'w') as fd:
        fd.write('\n'.join(options_lines))
