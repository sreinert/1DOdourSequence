import numpy as np
import soundfile as sf

def generate_white_noise(duration, sample_rate):
    # Number of samples is duration multiplied by the sample rate
    num_samples = int(duration * sample_rate)
    
    # Generate white noise values between -1 and 1
    noise = np.random.uniform(-1, 1, num_samples)
    
    return noise

# def generate_sound(duration, frequency, sample_rate = 16000):
#     num_samples = int(duration * sample_rate)
#     sound = np.ones(num_samples) * frequency
#     return sound

import numpy as np
import soundfile as sf

# Constants
FS = 44100  # Sampling frequency, typical for audio
DURATION = 5  # Duration of the tone in seconds
FREQ = 15000  # Frequency of the tone in Hz

# Generate time array
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

# Generate the pure tone of 6kHz
tone = np.sin(2 * np.pi * FREQ * t)

# Save as OGG file
sf.write('15kHz_tone.ogg', tone, FS)

    

# # Parameters
# duration = 10  # in seconds
# sample_rate = 16000  # 16kHz

# # Generate white noise
# # noise = generate_white_noise(duration, sample_rate)

# # Save to OGG file
# # sf.write('white_noise.ogg', noise, sample_rate)

# noise = generate_sound(duration, frequency=0.7)
# sf.write('sound07.ogg', noise, sample_rate)

# noise = generate_sound(duration, frequency=0.3)
# sf.write('sound03.ogg', noise, sample_rate)

# print("File 'white_noise.ogg' generated!")
