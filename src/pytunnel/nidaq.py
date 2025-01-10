from __future__ import division, print_function

import logging
import time

import PyDAQmx as daq
import numpy as np


def make_input(chan, diameter=None, constant_value=None, **kwargs):
    # create an angular encoder input if diameter is given
    if diameter is not None:
        input_chan = AngularVelocityEncoder(chan, diameter, **kwargs)

    # or a constant value fake channel
    elif constant_value is not None:
        input_chan = ConstantInput(constant_value)

    # otherwise create an analog channel
    else:
        input_chan = AnalogInput(chan, **kwargs)

    return input_chan


def make_output(*args, **kwargs):
    # alias for analog ouput for now
    return AnalogOutput(*args, **kwargs)


def make_trigger(chan, freq=None, duty_cycle=0.5, **kwargs):
    # create a counter trigger
    if freq is not None:
        trigger_chan = PulseGenerator(chan, freq, duty_cycle)

    # or a simple digital output line
    else:
        trigger_chan = DigitalOutput(chan)

    return trigger_chan, kwargs


class AnalogInput:

    def __init__(self, chan, min_value, max_value, threshold=None):
        self.chan = chan
        self.min_value = min_value
        self.max_value = max_value
        self.threshold = threshold
        self.task = daq.Task()
        self.task.CreateAIVoltageChan(
            chan, "", daq.DAQmx_Val_RSE, min_value, max_value,
            daq.DAQmx_Val_Volts, None
        )

    def read_float(self):
        value_box = daq.float64()
        self.task.ReadAnalogScalarF64(-1, daq.byref(value_box), None)
        value = value_box.value
        if self.threshold is not None:
            value = value > self.threshold
        return value
    
    def close(self):
        # Call the ClearTask method to release the port
        if self.task is not None:
            print('closing analog input')
            self.task.ClearTask()
            self.task = None


class ConstantInput:

    def __init__(self, constant_value):
        self.constant_value = constant_value

    def read_float(self):
        return self.constant_value
    
    def close(self):
        pass


class AnalogOutput:

    def __init__(self, chan, min_value, max_value, scale=1):
        self.chan = chan
        self.min_value = min_value
        self.max_value = max_value
        self.scale = scale

        logging.basicConfig()
        self.logger = logging.getLogger(__name__)

        self.task = daq.Task()
        self.task.CreateAOVoltageChan(
            chan, "", min_value, max_value, daq.DAQmx_Val_Volts, None
        )

    def write_float(self, value):
        value *= self.scale

        if value < self.min_value or value > self.max_value:
            self.logger.error('scaled value %f clipped before writing to %s',
                              value, self.chan)

        value = min(value, self.max_value)
        value = max(value, self.min_value)
        self.task.WriteAnalogScalarF64(1, -1, value, None)
        
    def close(self):
        # Call the ClearTask method to release the port
        if self.task is not None:
            print('closing analog output')
            self.task.ClearTask()
            self.task = None


class DigitalOutput:

    def __init__(self, chan):
        self.chan = chan
        self.task = daq.Task()
        self.task.CreateDOChan(chan, "", daq.DAQmx_Val_GroupByChannel)

    def write_uint(self, value):
        value_arr = np.array(value, dtype=np.uint8)
        self.task.WriteDigitalLines(
            1, 1, -1, daq.DAQmx_Val_GroupByChannel, value_arr, None, None
        )

    def start(self):
        self.write_uint(1)

    def stop(self):
        self.write_uint(0)
        
    def close(self):
        # Call the ClearTask method to release the port
        if self.task is not None:
            print('closing digital output')
            self.task.ClearTask()
            self.task = None


class PulseGenerator:

    def __init__(self,  chan, freq, duty_cycle):
        self.chan = chan
        self.task = daq.Task()
        self.task.CreateCOPulseChanFreq(
            chan, "", daq.DAQmx_Val_Hz, daq.DAQmx_Val_Low, 0.0, freq,
            duty_cycle
        )
        self.task.CfgImplicitTiming(daq.DAQmx_Val_ContSamps, 1000)

    def start(self):
        self.task.StartTask()

    def stop(self):
        self.task.StopTask()
        
    def close(self):
        # Call the ClearTask method to release the port
        if self.task is not None:
            self.task.ClearTask()
            self.task = None


class AngularEncoder:

    def __init__(self, chan, pulses_per_rev, error_value,
                 encoder_type=daq.DAQmx_Val_X4, units=daq.DAQmx_Val_Ticks):

        self.chan = chan
        self.error_value = error_value

        self.task = daq.Task()
        self.task.CreateCIAngEncoderChan(
            chan, "", encoder_type, False, 0, daq.DAQmx_Val_AHighBHigh,
            units, pulses_per_rev, 0, ""
        )
        self.task.StartTask()

    def read_float(self):
        value_box = daq.float64()
        self.task.ReadCounterScalarF64(-1, daq.byref(value_box), None)
        return value_box.value if value_box.value < self.error_value else 0

    def close(self):
        # Call the ClearTask method to release the port
        if self.task is not None:
            print('closing angular encoder')
            self.task.ClearTask()
            self.task = None

class AngularVelocityEncoder:

    def __init__(self, chan, diameter, threshold=None, gain=1, **kwargs):
        self.encoder = AngularEncoder(
            chan, units=daq.DAQmx_Val_Radians, **kwargs
        )
        self.diameter = diameter
        self.threshold = threshold
        self.gain = gain
        self.position = 0
        self.time = time.time()

    def read_float(self):
        new_position = self.encoder.read_float()
        new_time = time.time()

        # avoid crashing if read_float is called twice in a row
        # TODO fix calling code
        if new_time == self.time:
            return 0

        delta_radians = new_position - self.position
        delta_meter = delta_radians * self.diameter / 2
        delta_t = new_time - self.time
        speed = delta_meter / delta_t

        self.position = new_position
        self.time = new_time

        if self.threshold is not None and np.abs(speed) < self.threshold:
            speed = 0

        return speed * self.gain

    def close(self):
        # Call the ClearTask method to release the port
        if self.encoder.task is not None:
            self.encoder.task.ClearTask()
            self.encoder.task = None