from direct.showbase.ShowBase import ShowBase
# from panda3d.core import AudioManager, AudioSound, AudioSoundData
# import random

# app = ShowBase()

# def generate_white_noise(duration_sec, sample_rate=44100, amplitude=0.5):
#     # Calculate the number of samples required for the desired duration
#     num_samples = int(duration_sec * sample_rate)
    
#     # Generate random noise samples between -1.0 and 1.0
#     noise_samples = [random.uniform(-1.0, 1.0) * amplitude for _ in range(num_samples)]
    
#     # Create an AudioSoundData object with the noise samples
#     sound_data = AudioSoundData()
#     sound_data.setNumChannels(1)  # 1 channel for mono sound
#     sound_data.setSampleRate(sample_rate)
#     sound_data.setData(bytes(noise_samples))
    
#     return sound_data

# def play_sound(sound_data):
#     audio_mgr = AudioManager.createAudioManager()
#     sound = audio_mgr.getSound(sound_data)
#     sound.play()

# # Generate 5 seconds of white noise and play it
# duration_sec = 5
# white_noise_data = generate_white_noise(duration_sec)
# play_sound(white_noise_data)

# app.run()
from panda3d.core import AudioSound
def stopstart():
    if mySound.status() == AudioSound.PLAYING:
        mySound.stop()
    else:
        mySound.play()
        
base = ShowBase()
# mySound = base.loader.loadSfx("/home/masahiron/Downloads/file_example_OOG_1MG.ogg")
mySound = base.loader.loadSfx("/mnt/masahiron/swc-homes/src/srcMaNa_20230517_1DSequence/1DSequenceTaskPy/misc/white_noise.ogg")
base.accept("space", stopstart)
mySound.play()
base.run()

