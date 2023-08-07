#!/usr/bin/env python

from __future__ import division, print_function
import csv
import os

import logging
from pathlib import Path

import numpy as np
from direct.task import Task
from panda3d.core import CardMaker, ClockObject

from base_tunnel import BaseTunnel
from misc import default_main
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


class FlipTunnel:

    def __init__(self, tunnel, card, flip_sections, inputs, outputs,
                 sleep_time, test_mode, options):
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
        
        try:
            self.lock_corridor_reward = options['flip_tunnel']['lock_corridor_reward']
        except:
            self.lock_corridor_reward = True

        self.globalClock = ClockObject.getGlobalClock()
        
        self.total_forward_run_distance = 0
        
        # This will make the later analysis so much easier
        self.sample_i = 0

        
        # self.goals = [[0, 9], [36, 45], [54, 63]]
        try:
            self.goals = options['flip_tunnel']['goals']
        except:
            self.goals = [[0, 9]]
        self.goalNums = len(self.goals)
        try:
            self.currentGoal = options['flip_tunnel']['initial_goal']
        except:
            self.currentGoal = 0
        self.currentGoalIdx = 0
        
        try:
            manual_reward_with_space = options['flip_tunnel']['manual_reward_with_space']
        except:
            manual_reward_with_space = False

        self.ruleName = options['sequence_task']['rulename']

        self.isChallenged = False
        self.wasChallenged = False
        self.wasRewarded = False
        self.wasManuallyRewarded = False
        self.wasAssistRewarded = False
        
        try:
            self.punishBySpeedGain = options['flip_tunnel']['punishBySpeedGain']
        except:
            self.punishBySpeedGain = False

        self.isLogging = False

        self.triggeResetPosition = options['flip_tunnel']['length']
        try:
            self.triggeResetPositionStart = options['flip_tunnel']['margin_start']
            # self.reset_camera(self.triggeResetPositionStart)
        except:
            self.triggeResetPositionStart = 0
        
        self.flip_tunnel_options = options['flip_tunnel']
        self.flip_tunnel_options['corridor_len'] = options['flip_tunnel']['length'] - options['flip_tunnel']['margin_start']
        print('corridor length is ', self.flip_tunnel_options['corridor_len'])

        self.create_nidaq_controller(options)

        try:
            foldername = options['logger']['foldername']
            self.setup_logfile(foldername)
            self.isLogging = True
        except:
            self.isLogging = False

        if test_mode:
            self.tunnel.accept("space", self.spacePressed)
        if manual_reward_with_space:
            self.tunnel.accept("space", self.manualReward)

        # Add a task to check for the space bar being pressed
        self.tunnel.taskMgr.add(self.checkIfReward, "CheckIfRewardTask")
        
        # self.tunnel.taskMgr.add(self.checkIfPunished, "checkIfPunishedTask")

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

        if self.isNIDaq:
            self.tunnel.taskMgr.add(
                self.update_piezo_task, 'update_piezo_task'
            )

        if self.isLogging:
            self.tunnel.taskMgr.add(
                self.position_logging_task, 'position_logging_task'
            )
            # self.tunnel.taskMgr.add(
            #     self.event_logging_task, 'event_logging_task'
            # )
            
        if 'assisted_goals' in options['flip_tunnel']:
            print('Using assisted goals....')
            self.assisted_goals = options['flip_tunnel']['assisted_goals']
            self.tunnel.taskMgr.add(
                self.assist_reward_task, 'assist_reward_task'
            )
            
            if 'assist_reward_sec' in options['flip_tunnel']:
                self.assist_reward_sec = options['flip_tunnel']['assist_reward_sec']
            else:
                self.assist_reward_sec = 0.2
        
        if 'reward_sec' in options['flip_tunnel']:
            self.root_reward_sec = options['flip_tunnel']['reward_sec']
            self.reward_sec = options['flip_tunnel']['reward_sec']
        else:
            self.root_reward_sec = 0.2
            self.reward_sec = 0.2
            
        if 'manual_reward_sec' in options['flip_tunnel']:
            self.man_reward_sec = options['flip_tunnel']['manual_reward_sec']
        else:
            self.man_reward_sec = 0.1
            
        if 'wrong_reward_sec' in options['flip_tunnel']:
            self.wrong_reward_sec = options['flip_tunnel']['wrong_reward_sec']
            self.punishByRewardDecrease = True
        else:
            self.wrong_reward_sec = 0.1
            self.punishByRewardDecrease = False


    def setup_logfile(self, foldername):
        # Check if the directory exists, if not, create it
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        
        # Setup position log file
        position_filename = os.path.join(foldername, 'position_log.csv')
        position_file_exists = Path(position_filename).is_file()
        position_file = open(position_filename, 'a', newline='')
        self.position_writer = csv.writer(position_file)
        if not position_file_exists:
            self.position_writer.writerow(["Index", "Time", "Position", "TotalRunDistance", "Event"])

        # # Setup event log file
        # event_filename = os.path.join(foldername, 'event_log.csv')
        # event_file_exists = Path(event_filename).is_file()
        # event_file = open(event_filename, 'a', newline='')
        # self.event_writer = csv.writer(event_file)
        # if not event_file_exists:
        #     self.event_writer.writerow(["Time", "Event"])


    def spacePressed(self):
        print(self.tunnel.position)
        self.isChallenged = True
        
    def manualReward(self):
        self.wasManuallyRewarded = True
        self.triggerReward(length = self.man_reward_sec)

    # def checkIfPunished(self, task):
    #     # print('check if punished')
    #     if self.isChallenged and self.isAirpuff:
    #         print('checking is air puff is needed')
    #         if self.checkWithinPreviousOrCurrentGoal()!=True:
    #             print('triggering airpuff...')
    #             self.triggerAirpuff()
                
    #     if self.isChallenged and self.punishBySpeedGain:
    #         if self.checkWithinPreviousOrCurrentGoal()!=True:
    #             self.triggerGainDecrease()

    def checkIfPunished(self):
        # print('check if punished')
        if self.isAirpuff:
            # print('checking is air puff is needed')
            if self.checkWithinPreviousOrCurrentGoal()!=True:
                print('triggering airpuff...')
                self.triggerAirpuff()
                
        if self.punishBySpeedGain:
            if self.checkWithinPreviousOrCurrentGoal()!=True:
                self.triggerGainDecrease()
                
        if self.punishByRewardDecrease:
            if self.checkWithinPreviousOrCurrentGoal()!=True:
                self.reward_sec = self.wrong_reward_sec

    def checkIfReward(self, task):
        if self.isChallenged:
            self.checkIfPunished()
            print("licked, current pos: ", self.tunnel.position, " current goal ", self.goals[self.currentGoalIdx])
            self.isChallenged = False
            self.wasChallenged = True
            if self.checkWithinGoal():
                print('correct! Getting reward...')
                self.wasRewarded = True
                self.triggerReward(length=self.reward_sec)
                self.handleNextGoal()
                
        if self.ruleName in ['run-auto', 'protocol1_lv1'] :            
            pos = self.tunnel.position
            # position will be something like 99.0011 before being 9.0011, and it will cause bugs.
            # to prevent this, always subtract 90 if it is larger than 90
            if pos > self.flip_tunnel_options['length']:
                pos = pos - 90
            if  pos + self.total_forward_run_distance > self.currentGoal:
                # print(self.tunnel.position)
                # print(self.total_forward_run_distance)
                # print(self.currentGoal)
                self.wasRewarded = True
                self.triggerReward(length=self.reward_sec)
                self.handleNextGoal()
                
        return Task.cont
    
    def assist_reward_task(self, task):
        goals = self.assisted_goals[self.currentGoalIdx]
        position = self.tunnel.position
        if position > goals[0] and position < goals[1]:
            print('Getting reward with assist')
            self.wasAssistRewarded = True
            self.triggerReward(length=self.assist_reward_sec)
            self.handleNextGoal()
            
        return Task.cont
        

    def triggerReward(self, length=0.3):
        if self.isNIDaq:
            self.valveController.start()
            if self.lock_corridor_reward:
                time.sleep(length)
                self.valveController.stop()
            else:
                self.tunnel.taskMgr.doMethodLater(
                    length, self.stop_valve_task, 'stop_valve_task'
                )
        else:
            print(self.tunnel.position)
            time.sleep(length)
            print('reward is triggered')
            
    def triggerAirpuff(self):
        if self.isNIDaq:
            self.airpuffController.start()
            print('air puffed')
            if self.lock_corridor_reward:
                time.sleep(0.1)
                self.airpuffController.stop()
            else:
                self.tunnel.taskMgr.doMethodLater(
                    0.1, self.stop_airpuff_task, 'stop_airpuff_task'
                )
        else:
            time.sleep(0.1)
            print('air puffed')
            
    def triggerGainDecrease(self, scale=0.05, min_speed_gain=0.1, delay=3):
        speed_gain_before = self.speed_gain
        self.speed_gain = max(min_speed_gain, self.speed_gain - scale)
        
        self.tunnel.taskMgr.doMethodLater(
            delay, self.increaseSpeeGain, 'increaseSpeeGain'
        )
        print('speed gain decreased from {} to {}'.format(speed_gain_before, self.speed_gain))
        
    def increaseSpeeGain(self, task, scale):
        self.speed_gain = self.speed_gain + scale
        
    def checkAssistGoal(self):
        goals = self.assisted_goals[self.currentGoalIdx]
        position = self.tunnel.position
        print('checkAssistGoal {} > {} and {}'.format(position, goals[0], goals[1]))
        if position > goals[0] and position < goals[1]:
            print('true')
            return True
        return False

    def checkWithinGoal(self):
        if self.ruleName == 'sequence':
            goals = self.goals[self.currentGoalIdx]
            position = self.tunnel.position
            if position > goals[0] and position < goals[1]:
                return True
            return False
        elif self.ruleName in ['all']:
            position = self.tunnel.position
            for goals in self.goals:
                if position > goals[0] and position < goals[1]:
                    return True
            return False
        elif self.ruleName in ['protocol1_lv2']:
            position = self.tunnel.position
            if self.tunnel.position + self.total_forward_run_distance > self.currentGoal:
                for goals in self.goals:
                    if position > goals[0] and position < goals[1]:
                        
                        return True
            return False
        elif self.ruleName in ['run-lick']:
            print(self.tunnel.position + self.total_forward_run_distance,  self.currentGoal)
            if self.tunnel.position + self.total_forward_run_distance > self.currentGoal:
                return True
            return False
        
    def checkWithinPreviousOrCurrentGoal(self):
        if self.ruleName == 'sequence':
            position = self.tunnel.position
            previousGoalIdx = (self.currentGoalIdx + (self.goalNums-1)) % self.goalNums #This is equal to subtracting one
            for goalIdx in [previousGoalIdx, self.currentGoalIdx]:
                goals = self.goals[goalIdx]
                if position > goals[0] and position < goals[1]:
                    return True
            return False
        else:
            return True
        

    def handleNextGoal(self):
        if self.ruleName == 'sequence':
            self.currentGoalIdx = (self.currentGoalIdx + 1) % self.goalNums
            print('next goal is set to {}'.format(self.currentGoalIdx))
            self.reward_sec = self.root_reward_sec
        elif self.ruleName == 'run-auto' or self.ruleName == 'run-lick':
            self.currentGoal = self.currentGoal + np.random.randint(10) + self.flip_tunnel_options['reward_distance']
        elif self.ruleName in ['protocol1_lv1', 'protocol1_lv2']:
            self.currentGoal = self.currentGoal + self.flip_tunnel_options['reward_distance']
            print('next goal is set to {}'.format(self.currentGoal))

    def reset_tunnel_task(self, task):
        self.current_flip_sections = []
        for i, section in enumerate(self.flip_sections):
            section.reset()
            self.logger.info('section_id %d, new stim %d', i, section.stim_id)
        self.tunnel.freeze(False)
        self.tunnel.reset_camera(position=self.triggeResetPositionStart)
        self.total_forward_run_distance += self.flip_tunnel_options['corridor_len']
    
    def stop_valve_task(self, task):
        self.valveController.stop()
        
    def stop_airpuff_task(self, task):
        self.airpuffController.stop()
        
        
    def reset_tunnel2end_task(self, task):
        self.current_flip_sections = []
        for i, section in enumerate(self.flip_sections):
            section.reset()
            self.logger.info('section_id %d, new stim %d', i, section.stim_id)
        self.tunnel.freeze(False)
        self.tunnel.reset_camera(position=self.triggeResetPosition)
        self.total_forward_run_distance -= self.flip_tunnel_options['corridor_len']

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

        if self.tunnel.position > self.triggeResetPosition and not self.tunnel.frozen:
            self.tunnel.freeze(True)
            self.tunnel.taskMgr.doMethodLater(
                self.sleep_time, self.reset_tunnel_task, 'reset_tunnel_task'
            )
            
        if self.tunnel.position < self.triggeResetPositionStart and not self.tunnel.frozen:
            self.tunnel.freeze(True)
            self.tunnel.taskMgr.doMethodLater(
                self.sleep_time, self.reset_tunnel2end_task, 'reset_tunnel2end_task'
            )

        return Task.cont

    def create_nidaq_controller(self, options):
        if 'daqChannel' in options:
            import nidaq as nidaq
            self.valveController = nidaq.DigitalOutput(
                options['daqChannel']['valve1'])
            self.lickDetector = nidaq.AnalogInput(
                **options['daqChannel']['spout1'])
            
            if 'airpuff' in options['daqChannel']:
                self.airpuffController = nidaq.DigitalOutput(
                    options['daqChannel']['airpuff'])
                self.isAirpuff = True
            else:
                self.isAirpuff = False
            
            self.isNIDaq = True
        else:
            self.logger.warn("no daq channel specified, "
                             "using default channel 0")
            self.isNIDaq = False
            self.isAirpuff = False

    def update_inputs_task(self, task):
        speed = self.tunnel.speed
        if 'speed' in self.inputs:
            speed = self.inputs['speed'].read_float()

        self.logger.info("speed: %f", speed)
        self.tunnel.speed = speed

        return Task.cont

    def update_piezo_task(self, task):

        self.isChallenged = self.lickDetector.read_float()
        # print('self.isChallenged', self.isChallenged)
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
        


    def position_logging_task(self, task):
        if not hasattr(task, 'next_log_time'):
            task.next_log_time = self.globalClock.getFrameTime()

        current_time = self.globalClock.getFrameTime()

        if current_time < task.next_log_time:
            return Task.cont

        task.next_log_time += 1.0 / 60.0  # Schedule the next log in 1/60th of a second

        position = self.tunnel.position
        total_run_distance = self.total_forward_run_distance + position
        self.position_writer.writerow([self.sample_i, current_time, position, total_run_distance, ''])
        
        if self.wasChallenged:
            # print("challenged")
            self.position_writer.writerow([self.sample_i, current_time, -1, -1, "challenged"])
            self.wasChallenged = False

        if self.wasRewarded:
            self.position_writer.writerow([self.sample_i, current_time, -1, -1, "rewarded"])
            print("rewarded")                
            self.wasRewarded = False
            
        if self.wasManuallyRewarded:
            self.position_writer.writerow([self.sample_i, current_time, -1, -1, "manually-rewarded"])
            print("mannualy rewarded")
            self.wasManuallyRewarded = False
            
        if self.wasAssistRewarded:
            self.position_writer.writerow([self.sample_i, current_time, -1, -1, "assist-rewarded"])
            print("assist rewarded")
            self.wasAssistRewarded = False
            
        self.sample_i += 1

        return Task.cont

    # def event_logging_task(self, task):
    #     if self.wasChallenged:
    #         print('evemt logger was chalenged')
    #         print([self.shared_timestamp, "challenged"])
    #         self.position_writer.writerow([self.shared_timestamp, -1, "challenged"])
    #         self.event_writer.writerow([self.shared_timestamp, "challenged"])
    #         self.wasChallenged = False

    #     if self.wasRewarded:
    #         print('event logger was rewarded')
    #         self.position_writer.writerow([self.shared_timestamp, -1, "rewarded"])
    #         self.event_writer.writerow([self.shared_timestamp, "rewarded"])
    #         print([self.shared_timestamp, "rewarded"])
    #         self.wasRewarded = False

    #     return Task.cont