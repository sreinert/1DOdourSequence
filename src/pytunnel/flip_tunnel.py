#!/usr/bin/env python

from __future__ import division, print_function

import logging

import numpy as np
from direct.task import Task
from panda3d.core import CardMaker

from base_tunnel import BaseTunnel
from misc import default_main

import pytunnel.nidaq as nidaq
import time

class Flipper:

    def __init__(self, probas, max_noflip_trials):
        assert np.isclose(np.sum(probas), 1)
        self.probas = np.array(probas) / np.sum(probas)
        n_stim = len(probas)

        # define the number of transitions without flip, i.e same texture
        max_noflip_trials = np.atleast_1d(max_noflip_trials)
        if len(max_noflip_trials) != 1 and len(max_noflip_trials) != n_stim:
            raise ValueError(
                "'max_noflip_trials' should have 1 or {} elements."
                .format(n_stim)
            )
        self.max_noflip_trials = np.empty(n_stim)
        self.max_noflip_trials[...] = max_noflip_trials

        self.stim_trials = np.zeros(n_stim)
        self.noflip_trials = 0
        self.stim_id = None

    def __str__(self):
        return "stim_id: {}, noflip_trials: {}, stim_trials: {}".format(
            self.stim_id, self.noflip_trials, self.stim_trials
        )

    @property
    def n_stim(self):
        return len(self.probas)

    def next_stim_id(self):
        # randomly generate the next stimulus ID
        p = self.probas.copy()

        # avoid more than 'max_noflip_trials' trials with the same stimulus
        nnz_p = len(p.nonzero()[0])

        if self.stim_id is None:  # very first trial
            max_noflip = np.inf
        else:
            max_noflip = self.max_noflip_trials[self.stim_id]

        if nnz_p > 1 and self.noflip_trials >= max_noflip:
            p[self.stim_id] = 0
            p /= p.sum()

        sample = np.random.multinomial(1, p)
        next_stim_id = sample.nonzero()[0][0]

        # update counter of noflip trials
        if next_stim_id == self.stim_id:
            self.noflip_trials += 1
        else:
            self.noflip_trials = 0

        self.stim_id = next_stim_id
        self.stim_trials[next_stim_id] += 1

        return self.stim_id


class BlockFlipper:

    def __init__(self, probas, n_block):
        assert np.isclose(np.sum(probas), 1)
        self.probas = np.array(probas) / np.sum(probas)
        self.n_block = n_block

        self.stim_id = None
        self.cnt = 0

        counts = [p * n_block for p in probas]
        if any(not np.isclose(cr, int(cr)) for cr in counts):
            raise ValueError('Block size does not allow to draw whole number '
                             'samples for all stimulus types.')

        block = [np.full(int(c), i) for i, c in enumerate(counts)]
        self.block = np.concatenate(block)
        np.random.shuffle(self.block)

    def __str__(self):
        return "stim_id: {} ({} / {})".format(
            self.stim_id, self.cnt, self.n_block
        )

    @property
    def n_stim(self):
        return len(self.probas)

    def next_stim_id(self):
        self.stim_id = self.block[self.cnt]

        self.cnt += 1
        if self.cnt >= self.n_block:
            self.cnt = 0
            np.random.shuffle(self.block)

        return self.stim_id


class Follower:

    def __init__(self, flipper):
        self.flipper = flipper

    def __str__(self):
        return str(self.flipper) + ' (linked)'

    @property
    def n_stim(self):
        return self.flipper.n_stim

    def next_stim_id(self):
        return self.flipper.stim_id


class FlipSection:

    def __init__(self, section, stimulus_textures, stimulus_onset, triggers,
                 flipper):
        assert len(stimulus_textures) == len(triggers)
        assert len(stimulus_textures) == flipper.n_stim

        self.section = section
        self.onset = section.getPos()[1] - stimulus_onset
        self.offset = section.getPos()[1] + stimulus_onset

        self.neutral_texture = section.getTexture()
        self.stimulus_textures = stimulus_textures

        self.flipper = flipper
        self.triggers = triggers

        self.stim_id = None
        self.triggered = False
        self.reset()

    @property
    def n_stim(self):
        return len(self.stimulus_textures)

    def reset(self):
        self.section.setTexture(self.neutral_texture, 1)  # reset texture
        self.stim_id = self.flipper.next_stim_id()
        self.triggered = False

    def update(self, tunnel):
        # do nothing if outside of triggering zone
        if tunnel.position < self.onset or tunnel.position > self.offset:
            return False

        # do nothing if already triggered
        if self.triggered:
            return True

        self.triggered = True

        # change the texture for the current stimulus
        self.section.setTexture(self.stimulus_textures[self.stim_id], 1)

        # start a trigger, if any for the current stimulus
        for trigger in self.triggers[self.stim_id]:
            tunnel.taskMgr.add(
                trigger, 'start_trigger_task', sort=-20,
                extraArgs=[tunnel.position], appendTask=True
            )

        return True


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


class FlipTunnel:

    def __init__(self, tunnel, card, flip_sections, inputs, outputs,
                 sleep_time, test_mode):
        """_summary_

        Args:
            tunnel (_type_): _description_
            card (_type_): _description_
            flip_sections (_type_): _description_
            inputs (_type_): _description_
            outputs (tuple(key: AnalogOutput)): handles NIDAQ control, in this case is a reward valve. 
                                    .write_float method uses daq.Task.WriteAnalogScalarF64
            sleep_time (_type_): _description_
            test_mode (_type_): _description_
        """

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)

        self.tunnel = tunnel
        self.current_flip_sections = []
        self.flip_sections = flip_sections
        self.card = card
        self.sleep_time = sleep_time
        self.inputs = inputs
        self.outputs = outputs

        self.goalNums = 3
        self.currentGoal = 0
        self.goals = [[9, 18], [45, 54], [63, 72]]

        self.ruleName = 'sequence'

        # Add a task to check for the space bar being pressed
        self.tunnel.taskMgr.add(self.checkIfReward, "CheckIfRewardTask")

        for i, section in enumerate(self.flip_sections):
            self.logger.info('section_id %d, new stim %d', i, section.stim_id)

        # add a task to the tunnel to update walls according to the position
        self.tunnel.taskMgr.add(
            self.update_tunnel_task, 'update_tunnel_task', sort=5
        )

        # add an input task, with higher priority (executed before moving)
        self.tunnel.taskMgr.add(
            self.update_inputs_task, 'update_inputs_task', sort=-10
        )

        # add an output with lower priority (executed after updating tunnel)
        self.tunnel.taskMgr.add(
            self.update_outputs_task, 'update_outputs_task', sort=10
        )

        self.valveController = nidaq.DigitalOutput(options['daqChannel']['valve1'])

    def checkIfReward(self, task):
        if self.tunnel.isChallenged:
            print("Pressed, current position is", self.tunnel.position)
            self.tunnel.isChallenged = False
            if self.checkWithinGoal():
                print('correct! Getting reward...')
                self.triggerReward()
                self.handleNextGoal()
        return Task.cont

    def triggerReward(self):
        self.valveController.start()
        time.sleep(0.1)
        self.valveController.stop()
        return

    def checkWithinGoal(self):
        if self.ruleName == 'sequence':
            goals = self.goals[self.currentGoal]
            position = self.tunnel.position
            if position > goals[0] and position < goals[1]:
                return True
            return False
        elif self.ruleName == 'all':
            position = self.tunnel.position
            for goal in self.goals:
                if position > goals[0] and position < goals[1]:
                    return True
            return False

    def handleNextGoal(self):
        self.currentGoal = (self.currentGoal + 1) % self.goalNums
        print('next goal is set')

    def reset_tunnel_task(self, task):
        self.current_flip_sections = []
        for i, section in enumerate(self.flip_sections):
            section.reset()
            self.logger.info('section_id %d, new stim %d', i, section.stim_id)
        self.tunnel.freeze(False)
        self.tunnel.reset_camera()

    def update_tunnel_task(self, task):
        # update grating sections, if mouse is in their onset/offset part
        self.current_flip_sections = []
        card_color = [0, 0, 0, 1]

        for sid, section in enumerate(self.flip_sections):
            if section.update(self.tunnel):
                self.logger.info("section_id: %d, %s", sid, section.flipper)
                self.current_flip_sections.append(section)
                card_color = [1, 1, 1, 1]

        # change card color to indicate stimulus on
        self.card.setColor(*card_color)

        # end of trial update
        # at each end of trial (or tunnel), water reward is given
        # if self.tunnel.end_reached and not self.tunnel.frozen:
        #     self.tunnel.freeze(True)
        #     self.tunnel.taskMgr.doMethodLater(
        #         self.sleep_time, self.reset_tunnel_task, 'reset_tunnel_task'
        #     )

        # print(self.tunnel.position)

        if self.tunnel.position > 90 and not self.tunnel.frozen:
            self.tunnel.freeze(True)
            self.tunnel.taskMgr.doMethodLater(
                self.sleep_time, self.reset_tunnel_task, 'reset_tunnel_task'
            )

        return Task.cont

    def update_inputs_task(self, task):
        speed = self.tunnel.speed
        if 'speed' in self.inputs:
            speed = self.inputs['speed'].read_float()

        self.logger.info("speed: %f", speed)
        self.tunnel.speed = speed

        return Task.cont

    def update_outputs_task(self, task):
        if 'stim_id' in self.outputs:
            if self.current_flip_sections:
                if len(self.current_flip_sections) > 1:
                    self.logger.warn("multiple sections triggered, "
                                     "reporting stimulus ID of the first one")
                section = self.current_flip_sections[0]
                self.outputs['stim_id'].write_float(section.stim_id + 1)
            else:
                self.outputs['stim_id'].write_float(0)

        if 'position' in self.outputs:
            self.outputs['position'].write_float(self.tunnel.scaled_position)

        if 'speed' in self.outputs:
            self.outputs['speed'].write_float(self.tunnel.speed)

        return Task.cont

    def run(self):
        self.tunnel.run()


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


def make_flip_tunnel(tunnel_options, card_options, flip_tunnel_options,
                     walls_sequence, in_options, out_options, test_mode):

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
    tunnel = BaseTunnel(walls_textures, test_mode=test_mode, **tunnel_options)

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
        tunnel, card, flip_sections, inputs, outputs, sleep_time, test_mode
    )

    # append more triggers for open-loop sections
    for section, open_loop in zip(flip_sections, open_loop_options):
        if open_loop is None:
            continue
        append_openloop_task(flip_tunnel, section, **open_loop)

    return flip_tunnel


if __name__ == "__main__":
    options = default_main()
    no_input_speed = not ('speed' in options['inputs'])
    flip_tunnel = make_flip_tunnel(
        options['base_tunnel'], options['card'], options['flip_tunnel'],
        options['walls_sequence'], options['inputs'], options['outputs'],
        test_mode=no_input_speed
    )
    flip_tunnel.run()
