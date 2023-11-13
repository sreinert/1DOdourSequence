#!/usr/bin/env python
import logging
import numpy as np
from direct.task import Task
from panda3d.core import CardMaker

from base_tunnel import BaseTunnel
from misc import default_main

from flip_tunnel import BlockFlipper, FlipSection, FlipTunnel, Flipper, Follower


def make_trigger_task(tunnel, trigger_chan, delay=None, onset=None,
                      duration=None, extent=None):

    # force user to provide an extent or a duration
    if extent is None and duration is None:
        raise ValueError("'extent' or 'duration' must be defined")

    # immediate start if no delay nor onset
    if delay is None and onset is None:
        delay, onset = 0, 0

    logging.basicConfig()
    logger = logging.getLogger(__name__)

    # define a task function to stop the trigger after some time or space
    def stop_trigger_task(trigger_onset, task):
        time_done = duration is not None and task.time >= duration
        pos_done = (
            extent is not None and tunnel.position >= trigger_onset + extent
        )

        if time_done or pos_done:
            logger.info(
                "stop trigger (time: %s, position: %s)", time_done, pos_done
            )
            trigger_chan.stop()
            return Task.done

        return Task.cont

    # define a task function to start trigger after some time or space
    def start_trigger_task(stim_onset, task):
        time_done = delay is not None and task.time >= delay
        pos_done = (
            onset is not None and tunnel.position >= stim_onset + onset
        )

        if time_done or pos_done:
            logger.info("start trigger (time: %s, position: %s)",
                        time_done, pos_done)
            trigger_chan.start()
            tunnel.taskMgr.add(
                stop_trigger_task, 'stop_trigger_task', sort=-20,
                extraArgs=[tunnel.position], appendTask=True
            )
            return Task.done

        return Task.cont

    return start_trigger_task


def make_flip_sections(tunnel, walls_sequence, default_onset, io_module):
    """_summary_
    instantiate a list of FlipSection objects, possibly interlinked

    Args:
        tunnel (_type_): _description_
        walls_sequence (list): written in yaml file. Looks like below. Each item is either str
                               or dict.

        walls_sequence:
        - random_dots.png
        - random_dots.png
        - stimulus_textures: [grating1.jpg, grating2.jpg]
            triggers:
            - []
            - [{chan: Dev1/ctr2, delay: 0.16, duration: 1, duty_cycle: 0.4, freq: 20}, {chan: Dev2/port0/line0, onset: 3, duration: 0.2}]

        It would be
        {'stimulus_textures': ['grating1.jpg', 'grating2.jpg'], 
         'triggers': [
                [], 
                [{'chan': 'Dev1/ctr2', 'delay': 0.16, 'duration': 1, 'duty_cycle': 0.4, 'freq': 20}, 
                {'chan': 'Dev2/port0/line0', 'onset': 3, 'duration': 0.2}]
                ]}

        The length of 'stimulus_textures' should match the length of 'triggers'.


        default_onset (_type_): _description_
        io_module (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    load_texture = tunnel.loader.loadTexture
    flip_sections = []
    open_loop_options = []

    for walls, section in zip(walls_sequence, tunnel.sections):
        if isinstance(walls, str):
            continue

        assert isinstance(walls, dict)

        # load alternative textures
        # if not found, return default
        stimulus_onset = walls.get('stimulus_onset', default_onset)
        stimulus_textures = [
            load_texture(t) for t in walls['stimulus_textures']
        ]
        n_stim = len(stimulus_textures)

        # prepare triggers
        triggers = {i: [] for i in range(n_stim)}

        if 'triggers' in walls:

            # one trigger for one texture
            if len(triggers) != n_stim:
                raise ValueError("'triggers' should have {} elements."
                                 .format(n_stim))

            for i, trigger in enumerate(walls['triggers']):
                if not trigger:  # triggers may be [] or [{}] or [{},{}]
                    continue

                trigger = [trigger] if isinstance(trigger, dict) else trigger
                trigger_chan_n_kwargs = [
                    io_module.make_trigger(**trig) for trig in trigger
                ]
                triggers[i] = [
                    make_trigger_task(tunnel, chan, **kwargs)
                    for chan, kwargs in trigger_chan_n_kwargs
                ]

        # prepare flipping logic
        if 'linked_section' in walls:

            # prevent use of incompatible options
            random_options = ['probas', 'n_block', 'max_noflip_trials']
            if any(option in walls for option in random_options):
                raise ValueError(
                    "'linked_section' is not compatible with 'probas', "
                    "'n_block' and 'max_noflip_trials'."
                )

            linked_section = flip_sections[walls['linked_section']]
            if n_stim != linked_section.n_stim:
                raise ValueError(
                    'Number of stimulus textures between the section ({}) and '
                    'its linked section ({}) should match.'
                    .format(n_stim, linked_section.n_stim)
                )

            flipper = Follower(linked_section.flipper)

        else:
            probas = walls.get('probas', np.ones(n_stim) / n_stim)
            if len(probas) != n_stim:
                raise ValueError(
                    "'probas' should have {} elements.".format(n_stim)
                )
            if not np.isclose(np.sum(probas), 1):
                raise ValueError("Elements of 'probas' should sum to 1.")

            if 'n_block' in walls:
                if 'max_noflip_trials' in walls:
                    raise ValueError(
                        "'n_block' is not compatible with 'max_noflip_trials'."
                    )
                flipper = BlockFlipper(probas, walls['n_block'])

            else:
                max_noflip_trials = walls.get('max_noflip_trials', np.inf)
                flipper = Flipper(probas, max_noflip_trials)

        # instantiate the new flip section and add it to the list
        new_flip_section = FlipSection(
            section, stimulus_textures, stimulus_onset, triggers, flipper
        )

        flip_sections.append(new_flip_section)
        open_loop_options.append(walls.get('open_loop'))

    return flip_sections, open_loop_options


def make_card(tunnel, size, position):
    """create a small dark patch to display stimulus onset"""

    card_width, card_height = size
    card_x, card_y = position

    cm = CardMaker('card')
    card = tunnel.render2d.attachNewNode(cm.generate())
    card.setScale(card_width, 1, card_height)
    card.setPos(card_x - card_width / 2, 0, card_y - card_height / 2)
    card.setColor(0, 0, 0, 1)
    return card


def append_openloop_task(flip_tunnel, flip_section, speed, duration):

    logging.basicConfig()
    logger = logging.getLogger(__name__)

    def stop_openloop_task():
        logger.info("stop open-loop")
        flip_tunnel.tunnel.taskMgr.add(
            flip_tunnel.update_inputs_task, 'update_inputs_task', sort=-10
        )
        return Task.done

    def start_openloop_task(stim_onset, task):
        logger.info("start open-loop")
        flip_tunnel.tunnel.taskMgr.remove('update_inputs_task')
        flip_tunnel.tunnel.taskMgr.doMethodLater(
            duration, stop_openloop_task, 'stop_openloop_task',
            extraArgs=[]
        )
        flip_tunnel.tunnel.speed = speed / flip_tunnel.tunnel.speed_gain
        return Task.done

    for k in flip_section.triggers.keys():
        flip_section.triggers[k].append(start_openloop_task)


def make_flip_tunnel(options, test_mode):

    tunnel_options = options['base_tunnel']
    card_options = options['card']
    flip_tunnel_options = options['flip_tunnel']
    walls_sequence = options['walls_sequence']
    in_options = options['inputs']
    out_options = options['outputs']

    neutral_texture = flip_tunnel_options['neutral_texture']
    default_onset = flip_tunnel_options['stimulus_onset']
    sleep_time = flip_tunnel_options['sleep_time']
    io_module = __import__(flip_tunnel_options['io_module'])

    # create base tunnel
    walls_textures = [
        walls if isinstance(walls, str) else
        walls.pop('neutral_texture', neutral_texture)
        for walls in walls_sequence
    ]
    tunnel = BaseTunnel(walls_textures, test_mode=test_mode, options=options,
                        ** tunnel_options)

    # create card to display stimulus onset
    card = make_card(tunnel, **card_options)

    # create objects managing flipping sections
    flip_sections, open_loop_options = make_flip_sections(
        tunnel, walls_sequence, default_onset, io_module
    )

    # create inputs/outputs channels
    inputs = {k: io_module.make_input(**v) for k, v in in_options.items()}
    outputs = {k: io_module.make_output(**v) for k, v in out_options.items()}

    # create flip tunnel object
    flip_tunnel = FlipTunnel(
        tunnel, card, flip_sections, inputs, outputs, sleep_time, test_mode, options
    )

    # append more triggers for open-loop sections
    for section, open_loop in zip(flip_sections, open_loop_options):
        if open_loop is None:
            continue
        append_openloop_task(flip_tunnel, section, **open_loop)

    return flip_tunnel
