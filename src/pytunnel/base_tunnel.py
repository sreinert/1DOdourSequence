#!/usr/bin/env python

from __future__ import division, print_function

import sys
import logging

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Camera, PerspectiveLens, NodePath, WindowProperties

from misc import default_main


def split_display(cam1, cam2, fov, fov_shift=None, dual_monitor=False):
    print('split_display')
    # display the first camera on the left part of the screen
    
    dr = cam1.node().getDisplayRegion(0)
    if not dual_monitor:
        dr.setDimensions(0, 0.5, 0, 1)

    # create a second camera and display it on half right of the window
    # dr2 = cam2.node().getDisplayRegion(0)
    # dr2.setDimensions(0.5, 1, 0, 1)

    # turn the cameras so that they look on each side
    if fov_shift is None:
        fov_shift = fov / 2

    cam1.setHpr(fov_shift, 0, 0)
    cam2.setHpr(-fov_shift, 0, 0)

    # adjust their field of view with a new lens
    lens = PerspectiveLens()
    lens.setAspectRatio(dr.getPixelWidth() / dr.getPixelHeight())
    lens.setFov(fov)
    cam1.node().setLens(lens)
    cam2.node().setLens(lens)


class BaseTunnel(ShowBase):

    def __init__(self, walls_textures, speed_gain, test_mode, options, wall_model, wall_length, wall_spacing,
                 bg_color=(0, 0, 0, 0), eye_fov=None, speed_gain_step=0.05):

        ShowBase.__init__(self)

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)

        self.wall_spacing = wall_spacing

        self.disableMouse()
        self.setBackgroundColor(*bg_color)
        self.camera.setPos(0, options['flip_tunnel']['margin_start'], 4)
        
        print(options['monitor'])
        print(options['monitor']['monitor1']['width'])
        print(options['monitor']['monitor2']['width'])
        print(options['monitor']['monitor1']['width'] + options['monitor']['monitor2']['width'])


        if 'monitor' in options and 'dual_monitor' in options['monitor'] and options['monitor']['dual_monitor']:
            props = WindowProperties()
            props.setSize(options['monitor']['monitor2']['width'], options['monitor']['monitor2']['height'])
            props.setOrigin(int(options['monitor']['monitor1']['width']), 0)

            self.firstWindow = self.openWindow(
                props=props, makeCamera=False)
            dr = self.firstWindow.makeDisplayRegion()
            self.cam1 = self.makeCamera(self.firstWindow)

            props.setSize(options['monitor']['monitor3']['width'], options['monitor']['monitor3']['height'])
            props.setOrigin(options['monitor']['monitor1']['width'] + options['monitor']['monitor2']['width'], 0)
            self.secondWindow = self.openWindow(
                props=props, makeCamera=False)
            dr = self.secondWindow.makeDisplayRegion()
            self.cam2 = self.makeCamera(self.secondWindow)

            if eye_fov is not None:
                split_display(self.cam1, self.cam2, **eye_fov, dual_monitor=True)
        else:
            # split the window to display 2 cameras
            if eye_fov is not None:
                # create a second camera
                self.cam2 = NodePath(Camera('cam2'))
                self.win.makeDisplayRegion().setCamera(self.cam2)

                # same position as first camera
                self.cam.reparentTo(self.camera)
                self.cam2.reparentTo(self.camera)

                # adjust display region, fov and angle
                split_display(self.cam, self.cam2, **eye_fov)

        self.speed_gain = speed_gain
        self.speed = 0

        # load tunnel sections with their textures
        self.sections = []
        for i, texture_path in enumerate(walls_textures):
            texture = self.loader.loadTexture(texture_path)
            self.sections.append(self.loader.loadModel(
                wall_model))
            self.sections[-1].reparentTo(self.render)
            self.sections[-1].setTexture(texture, 1)
            self.sections[-1].setScale(2,
                                       wall_length, 2)
            self.sections[-1].setPos(0, i * wall_spacing, 4)

        # add shortcuts for the test-mode
        if test_mode:
            self.accept('arrow_up', setattr, [self, 'speed', 1])
            self.accept('arrow_up-up', setattr, [self, 'speed', 0])
            self.accept('arrow_down', setattr, [self, 'speed', -1])
            self.accept('arrow_down-up', setattr, [self, 'speed', 0])

        # add shortcuts to increase/decrease speed gain
        self.accept('m', self.adjust_speed_gain, [speed_gain_step])
        self.accept('n', self.adjust_speed_gain, [-speed_gain_step])

        # add a shortcut to exit
        self.accept('escape', sys.exit)

        self.taskMgr.add(self.move_camera_task, 'move_camera_task')

    @property
    def length(self):
        return self.wall_spacing * len(self.sections)

    @property
    def position(self):
        return self.camera.getPos()[1]

    @property
    def scaled_position(self):
        return self.position / self.length

    @property
    def end_reached(self):
        return self.position >= self.length

    @property
    def frozen(self):
        return not self.taskMgr.hasTaskNamed('move_camera_task')

    def freeze(self, value):
        if value:
            self.taskMgr.remove('move_camera_task')
            self.logger.info("tunnel frozen")
        elif self.frozen:
            self.taskMgr.add(self.move_camera_task, 'move_camera_task')
            self.logger.info("tunnel unfrozen")

    def adjust_speed_gain(self, scale):
        self.speed_gain = max(0, self.speed_gain + scale)
        print('new speed gain: {}'.format(self.speed_gain))

    def move_camera_task(self, task):
        new_pos = self.camera.getY() + self.speed * self.speed_gain
        new_pos = min(max(new_pos, 0), self.length)  # stay within tunnel
        self.camera.setY(new_pos)
        self.logger.info("position: %f, speed: %f, speed gain: %f",
                         self.position, self.speed, self.speed_gain)
        return Task.cont

    def reset_camera(self, position=0):
        z_pos = self.camera.getPos()[2]
        self.camera.setPos(0, position, z_pos)


if __name__ == "__main__":
    options = default_main()
    tunnel = BaseTunnel(test_mode=True, **options)
    tunnel.run()
