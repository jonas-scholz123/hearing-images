import os
import imageio
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.io import wavfile
import pandas as pd

""" TODO: logarithmic spacing"""


class Hilbert():
    ''' Class to hold reusable values, simply for optimisation reasons'''

    def __init__(self, max_freq, p, N):
        self.max_freq = max_freq
        self.hilbert_curve = HilbertCurve(p, N)
        self.max_dist = self.hilbert_curve.distance_from_coordinates([2**p - 1, 0])
        self.logspace = np.logspace(np.log2(1), np.log2(self.max_freq), self.max_dist, base = 2)
        return
    def coord_to_freq(self, x, y):
        ''' maps coordinates x, y to frequency using pseudo-hilbert curve.
         Based on logarithmic spacing of audible frequencies.'''
        dist = self.hilbert_curve.distance_from_coordinates([x, y])
        assert(dist <= self.max_dist), "x = " + str(x) + " y = " + str(y)
        return self.logspace[dist - 1]

    def brightness_to_volume(self, img, x, y, max_vol):
        ''' maps brightness of pixel to a specific volume'''
        # TODO: Check if mapping should be linear, maybe squared like intensity?
        max_brightness = 255
        #return img[y][x]/max_brightness * max_vol # USE THIS FOR 0 VOLUME = WHITE
        return -(img[y][x] - 255)/max_brightness * max_vol #use this for 0 volume = black


#!/usr/bin/python
# based on : www.daniweb.com/code/snippet263775.html

def gen_partial_sound_wave(volume, freq, sample_rate, duration = 10000):
    #duration in milliseconds

    audio = []
    one_period = []

    num_samples = 1/freq * sample_rate
    samples = np.array(range(int(num_samples)))

    one_period = volume * np.sin(2* np.pi * freq * samples/sample_rate)

    assert len(one_period) != 0
    return np.array(one_period)

def add_waves(waves):
    max_period = max([len(w) for w in waves])
    combined = np.zeros(max_period)
    for wave in waves:
        combined += np.resize(wave, max_period)

    return combined/max(combined)

def save_wav(audio, file_name):
    # Open up a wav file
    wav_file =  f.open(file_name,"w")

    # wav params
    nchannels = 1

    sampwidth = 2

    nframes = len(audio)

    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

    wav_file.close()

    return


def converter(image_path, method):
    # Audio will contain a long list of samples (i.e. floating point numbers describing the
    # waveform).  If you were working with a very long sound you'd want to stream this to
    # disk instead of buffering it all in memory list this.  But most sounds will fit in
    # memory.

    if __name__ == "__main__":
        #if called here access folders, for testing
        image_name = "batman.png"
        image_loc = "./images/"
        image_path = image_loc + image_name
        #if not, then use supplied image path

    audio_loc = "./audio/"

    im = imageio.imread(image_path, ignoregamma = True, as_gray = True)
    #df = pd.DataFrame(im)
    #pd.set_option('display.max_columns', 256)
    #pd.set_option('display.max_rows', 256)
    #df

    image_size_x = len(im)
    image_size_y = len(im[0])

    #check that image is square
    assert image_size_x == image_size_y
    assert np.log2(image_size_x).is_integer()

    p = int(np.log2(image_size_x)) # 2^(2p) pixels in image
    N = 2 # number of dimensions, for images always 2

    waves = []
    sample_rate = 44100.0
    i = 0
    max_prints = 5
    tone_length = 5 # in seconds
    max_freq = 18000

    if method == "hilbert":
        hilbert = Hilbert(max_freq, p, N)
        for y, row in enumerate(im):
            for x, pixel in enumerate(row):
                i += 1
                volume = hilbert.brightness_to_volume(im, x, y, 1)
                if volume == 0: # no need to calculate zero amplitude ## TODO: why does this matter
                    continue
                freq = hilbert.coord_to_freq(x, y)
                if freq == 0:
                    continue
                wave = gen_partial_sound_wave(volume, freq, sample_rate)
                waves.append(wave)


    combined = add_waves(waves)
    audio = np.resize(combined, int(sample_rate*tone_length))
    audio_path = audio_loc + image_path.split("/")[-1].split(".")[0] + ".wav"
    wavfile.write(audio_path, int(sample_rate), audio)
    return os.path.abspath(audio_path)

if __name__ == "__main__":
    converter("", "hilbert")
