from __future__ import division, print_function

from math import fabs


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

    default_value = 0.1

    def __init__(self, chan, min_value, max_value, threshold=None):
        self.chan = chan
        self.min_value = min_value
        self.max_value = max_value
        self.threshold = None

    def read_float(self):
        value = self.default_value
        if self.threshold is not None:
            value = value > self.threshold
        return value
    
    def close(self):
        pass


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

    def write_float(self, value):
        value *= self.scale

        if value < self.min_value or value > self.max_value:
            print('bad scaled value {} for {}'.format(value, self.chan))

        print('write {} to {}'.format(value, self.chan))
        
    def close(self):
        pass


class DigitalOutput:

    def __init__(self, chan):
        self.chan = chan

    def write_uint(self, value):
        print('write {} to {}'.format(value, self.chan))

    def start(self):
        print('start {}'.format(self.chan))

    def stop(self):
        print('stop {}'.format(self.chan))
        
    def close(self):
        pass


class PulseGenerator:

    def __init__(self,  chan, freq, duty_cycle):
        self.chan = chan

    def start(self):
        print('start {}'.format(self.chan))

    def stop(self):
        print('stop {}'.format(self.chan))
        
    def close(self):
        pass


class AngularEncoder:

    def __init__(self, chan, *args, **kwargs):
        self.chan = chan
        self.count = 0

    def read_float(self):
        self.count += 10
        return self.count
    
    def close(self):
        pass


class AngularVelocityEncoder:

    default_value = 0.1

    def __init__(self, chan, diameter, threshold=None, gain=1, **kwargs):
        self.chan = chan
        self.diameter = diameter
        self.threshold = threshold
        self.gain = gain

    def read_float(self):
        value = self.default_value
        if self.threshold is not None and fabs(value) < self.threshold:
            value = 0
        return value * self.gain

    def close(self):
        pass