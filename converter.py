import os
import imageio
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.io import wavfile
import PIL
import numpy.fft as fft

""" TODO: logarithmic spacing"""


class Converter():

    def __init__(self, method, image_path, max_freq = 18000, min_freq = 30, im = None, resolution = 16, tone_length = 1):
        self.method = method
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.null_colour = "black"
        self.sample_rate = 44100.0
        self.max_vol = 1
        self.tone_length = tone_length # in seconds
        self.audio_loc = "./audio/"
        self.resolution = resolution
        self.image_path = image_path
        self.force_null = False #normalises colour to have lowest brightness = 0
        if image_path:
            self.audio_path = self.audio_loc + image_path.split("/")[-1].split(".")[0] + ".wav"
            self.im = imageio.imread(image_path, ignoregamma = True, as_gray = True)
        else:
            self.im = im

        self.image_size_x = len(self.im)
        self.image_size_y = len(self.im[0])
        if not self.is_right_format() and self.image_path:
            self.reformat()


        if self.method == "hilbert":
            #assert self.image_size_x == self.image_size_y, "uploaded non-square image"
            #assert np.log2(self.image_size_x).is_integer()
            p = int(np.log2(self.image_size_x)) # 2^(2p) pixels in image
            N = 2 # number of dimensions, for images always 2
            self.hilbert_curve = HilbertCurve(p, N)
            self.max_dist = self.hilbert_curve.distance_from_coordinates([2**p - 1, 0])

        if self.method == "snake":
            self.max_dist = self.image_size_x * self.image_size_y

        self.logspace = np.logspace(np.log2(min_freq), np.log2(self.max_freq), self.max_dist, base = 2)
        self.phase = {freq: 0 for freq in self.logspace}
        self.prev_volume = {vol: 0 for vol in self.logspace}


        return

    def set_audio_path(self, audio_path):
        self.audio_loc = audio_path
        return

    def set_null_colour(self, colour):
        self.null_colour = colour
        return

    def is_right_format(self):
        return self.image_size_x == self.image_size_y and type(self.im[0][0]) == np.float

    def reformat(self):
        #makes image have the correct format
        im = PIL.Image.open(self.image_path)
        im_resized = im.resize((self.resolution, self.resolution), PIL.Image.ANTIALIAS)
        new_name = ".".join(self.image_path.split(".")[0:-1]) + "_reduced.png"
        im_resized.save(new_name)
        self.im = imageio.imread(new_name, ignoregamma = True, as_gray = True)
        self.image_size_x = len(self.im)
        self.image_size_y = len(self.im[0])




        #if self.image_size_x != self.image_size_y:
            #return


    def coord_to_freq(self, x, y):
        ''' maps coordinates x, y to frequency using pseudo-hilbert curve.
         Based on logarithmic spacing of audible frequencies.'''
        if self.method == "hilbert":
            return self.get_hilbert_freq(x, y)
        elif self.method == "snake":
            return self.get_snake_freq(x, y)

    def get_snake_freq(self, x, y):
        dist = y*self.image_size_x + x
        assert(dist <= self.max_dist), "x = " + str(x) + " y = " + str(y)
        return self.logspace[dist - 1]

    def get_hilbert_freq(self, x, y):
        dist = self.hilbert_curve.distance_from_coordinates([x, y])
        assert(dist <= self.max_dist), "x = " + str(x) + " y = " + str(y)
        return self.logspace[dist - 1]

    def brightness_to_volume(self, x, y):
        ''' maps brightness of pixel to a specific volume'''
        # TODO: Check if mapping should be linear, maybe squared like intensity?
        max_brightness = 255
        if self.null_colour == "white":
            return self.im[y][x]/max_brightness * self.max_vol # USE THIS FOR 0 VOLUME = WHITE
        else:
            return -(self.im[y][x] - 255)/max_brightness * self.max_vol #use this for 0 volume = black

#    def convert(self):
#        # Audio will contain a long list of samples (i.e. floating point numbers describing the
#        # waveform).  If you were working with a very long sound you'd want to stream this to
#        # disk instead of buffering it all in memory list this.  But most sounds will fit in
#        # memory.
#
#        waves = []
#        i = 0
#        max_prints = 5
#        max_freq = 18000
#
#        if self.method != "spectrogram":
#            for y, row in enumerate(self.im):
#                if y % 2 != 0:continue
#                self.progress = 80 * y/self.image_size_y #80% of time taken by this
#                for x, pixel in enumerate(row):
#                    if x % 2 != 0:continue
#                    i += 1
#                    #print(i)
#                    volume = self.brightness_to_volume(x, y)
#                    if volume == 0: # no need to calculate zero amplitude ## TODO: why does this matter
#                        continue
#                    freq = self.coord_to_freq(x, y)
#                    if freq == 0:
#                        continue
#                    wave = self.gen_partial_sound_wave(volume, freq)
#                    waves.append(wave)
#
#            combined = self.add_waves(waves)
#            audio = combined
#            self.audio = np.resize(combined, int(self.sample_rate*self.tone_length))
#        return self.audio

    def f2i(self, f):
        return int(self.tone_length * f)

    def complex_entry(self, mag, angle):
        return mag * np.exp(1j*angle)

    def convert(self):
        # Audio will contain a long list of samples (i.e. floating point numbers describing the
        # waveform).  If you were working with a very long sound you'd want to stream this to
        # disk instead of buffering it all in memory list this.  But most sounds will fit in
        # memory.

        spectrum = np.zeros(int(self.sample_rate * self.tone_length), dtype = np.csingle)
        i = 0
        max_prints = 5
        max_freq = 18000
        lowest_vol = 1

        if self.method != "spectrogram":
            for y, row in enumerate(self.im):
                #if y % 2 != 0:continue
                self.progress = 80 * y/self.image_size_y #80% of time taken by this
                for x, pixel in enumerate(row):
                    #if x % 2 != 0:continue
                    i += 1
                    #print(i)
                    initial_volume = self.brightness_to_volume(x, y)
                    freq = self.coord_to_freq(x, y)
                    self.phase[freq] = 2* np.pi * freq * self.tone_length + self.phase[freq]


                    volume = max(min(abs(self.prev_volume[freq] - initial_volume), 1), 0.01) #volume is driven by how much pixel changed, at least 0.1, at most 1
                    spectrum[self.f2i(freq)] = self.complex_entry(volume, self.phase[freq])

                    lowest_vol = min(volume, lowest_vol)
                    self.prev_volume[freq] = initial_volume

        if self.force_null:
            spectrum -= lowest_vol
        self.audio = np.real(fft.ifft(spectrum, self.tone_length * self.sample_rate))
        self.audio = self.audio/max(self.audio) #normalise

        return self.audio


    def rgb2gray(self):
        self.im = np.dot(self.im[...,:3], [0.2989, 0.5870, 0.1140])
        return

    def save_audio(self, audio = None):
        audio_path = self.audio_loc + self.image_path.split("/")[-1].split(".")[0] + ".wav"
        if not audio:
            wavfile.write(self.audio_path, int(self.sample_rate), self.audio)
        else:
            wavfile.write(self.audio_path, int(self.sample_rate), audio)
        return os.path.abspath(audio_path)

    def gen_partial_sound_wave(self, initial_volume, freq, duration = 10000):
        #duration in milliseconds

        audio = []
        one_period = []

        num_samples = 1/freq * self.sample_rate
        num_samples = self.tone_length * self.sample_rate
        samples = np.array(range(int(num_samples)))

        #if self.prev_volume[freq] == volume: return np.zeros(len(samples)) # emphasize changes

        volume = max(min(abs(self.prev_volume[freq] - initial_volume), 1), 0.01) #volume is driven by how much pixel changed, at least 0.1, at most 1
        self.prev_volume[freq] = initial_volume

        one_period = volume * np.sin(2* np.pi * freq * samples/self.sample_rate + self.phase[freq])
        #print(freq)
        #if freq == 1787.3196890091187:
            #print(volume)
            #print("first : ", one_period[0])
            #print("last : ", one_period[-1])
            #print(self.phase[1787.3196890091187])

        # save phase
        self.phase[freq] = 2* np.pi * freq * samples[-1]/self.sample_rate + self.phase[freq]

        assert len(one_period) != 0
        return one_period

    def add_waves(self, waves):
        max_period = max([len(w) for w in waves])
        combined = np.zeros(max_period)
        for wave in waves:
            combined += np.resize(wave, max_period)

        return combined/max(combined)

    def set_phase(self, phase):
        self.phase = phase

    def set_prev_vol(self, vol):
        self.prev_volume = vol

    def save_wav(self, audio, file_name):
        # Open up a wav file
        #wav_file =  open(file_name,"w")
#
        ## wav params
        #nchannels = 1
#
        #sampwidth = 2
#
        #nframes = len(audio)
#
        ## 44100 is the industry standard sample rate - CD quality.  If you need to
        ## save on file size you can adjust it downwards. The stanard for low quality
        ## is 8000 or 8kHz.
        #comptype = "NONE"
        #compname = "not compressed"
        #wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
#
        ## WAV files here are using short, 16 bit, signed integers for the
        ## sample size.  So we multiply the floating point data we have by 32767, the
        ## maximum value for a short integer.  NOTE: It is theortically possible to
        ## use the floating point -1.0 to 1.0 data directly in a WAV file but not
        ## obvious how to do that using the wave module in python.
        #for sample in audio:
        #    wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
#
        #wav_file.close()
        wavfile.write(file_name, int(self.sample_rate), audio)

        return



#!/usr/bin/python
# based on : www.daniweb.com/code/snippet263775.html

''' TODO: DONT NORMALISE, JUST DIVIDE BY RESOLUTION^2'''
#%%
if __name__ == "__main__":
    #if called here access folders, for testing

    waves = []
    sample_rate = 44100.0
    i = 0
    max_prints = 5
    tone_length = 5 # in seconds
    max_freq = 18000

    image_name = "testim.png"
    image_loc = "./images/"
    image_path = image_loc + image_name

    converter = Converter(method = "hilbert", image_path = image_path, tone_length = tone_length)
    converter.convert()
    converter.save_wav(converter.audio, converter.audio_path)

#%%
if __name__ == "__main__":
    reader = imageio.get_reader('./images/full_car_vid.mp4')
    vid_fps = reader.get_meta_data()['fps']
    vid_length = reader.get_length()
    print(vid_fps)
    wanted_fps = 25
    frame_spacing = int(vid_fps/wanted_fps)
    j = 0
    audio = []
    phase = None
    images = []

    for im in reader:
        images.append(im)

    for im in images[0::frame_spacing]:
        print("frame ", j, " out of ", int(len(images)/frame_spacing))
        j += 1

        converter = Converter(method = "hilbert", image_path = False, im = im)
        if phase:
            converter.set_phase(phase)
            converter.set_prev_vol(vol)
        converter.tone_length = 1/wanted_fps
        converter.set_audio_path(converter.audio_loc + "car_vid.wav")
        converter.rgb2gray()
        converter.convert()
        phase = converter.phase
        vol = converter.prev_volume
        audio.append(converter.audio)
        #print("phase: ", converter.phase)
        #print("deltas: ", [converter.phase - prev_phase[k] for k in converter.phase.keys()])
        #prev_phase = converter.phase

    audio = np.concatenate(audio)

    converter.save_wav(audio, "car_audio.wav")

    #image_name = "batman.png"
    #image_loc = "./images/"
    #image_path = image_loc + image_name
    #converter = Converter(method = "hilbert", image_path = image_path)
    #wave1 = converter.gen_partial_sound_wave(1, freq = 100)
    #wave2 = converter.gen_partial_sound_wave(0.4, freq = 5000)
    #wave3 = converter.gen_partial_sound_wave(0.7, freq = 1000)
    #waves = [wave1, wave2, wave3]
    #audio = converter.add_waves(waves)
    #audio = np.resize(audio, int(44100*5))
    #converter.save_wav(audio, "test.wav")
    #wavfile.write("test.wav", 44100, audio)
    #from playsound import playsound
    #playsound("test.wav")
