import os
import imageio
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.io import wavfile
import PIL
import numpy.fft as fft

class Converter():

    def __init__(self, method, image_path, max_freq = 18000, min_freq = 30, im = None,
                resolution = 16, tone_length = 1):

        # Paramters of Conversion
        self.method = method

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.null_colour = "black" #background colour
        self.force_null = False #normalises colour to have lowest brightness = 0
        self.sample_rate = 44100.0
        self.max_vol = 1
        self.tone_length = tone_length # in seconds

        self.resolution = resolution

        self.image_path = image_path # for reading image (uploaded)
        self.audio_loc = "./audio/" #  for saving audio


        # Doesn't have image path when live-streaming
        if image_path:
            # defines name for audio file
            self.audio_path = self.audio_loc + image_path.split("/")[-1].split(".")[0] + ".wav"
            #reads image as greyscale
            self.im = imageio.imread(image_path, ignoregamma = True, as_gray = True)
        else:
            #When live streaming, images are fed in from camera
            self.im = im

        # Read image size
        self.image_size_x = len(self.im)
        self.image_size_y = len(self.im[0])

        #If still image and it's not the right format, reformat
        #For livestream images supplied are already right format, no need to check
        if not self.is_right_format() and self.image_path:
            self.reformat()

        # If hilbert is used, initialise the hilbert curve from library
        if self.method == "hilbert":
            #requires values p, N for initialisation

            #variable resolution hilbert curve
            p = int(np.log2(self.image_size_x)) # 2^(2p) pixels in image
            N = 2 # number of dimensions, for images always 2

            #initialise curve
            self.hilbert_curve = HilbertCurve(p, N)
            #get maximum distance from starting point on hilbert curve, this
            #acts as a reference point for conversion
            self.max_dist = self.hilbert_curve.distance_from_coordinates([2**p - 1, 0])

        if self.method == "l2r":
            self.max_dist = self.image_size_x * self.image_size_y

        # logspace is needed because tones are logarithmically spaced for human hearing
        self.logspace = np.logspace(np.log2(min_freq), np.log2(self.max_freq), self.max_dist, base = 2)

        # Phase of sin-waves are initially 0, keep track of this for video,
        # so that waves are continuous when new frame is calculated
        self.phase = {freq: 0 for freq in self.logspace}

        # Keeps track of previous volume to emphasize change
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
        #makes image have the correct format by saving image and using PIL library
        im = PIL.Image.open(self.image_path)
        im_resized = im.resize((self.resolution, self.resolution), PIL.Image.ANTIALIAS)
        new_name = ".".join(self.image_path.split(".")[0:-1]) + "_reduced.png"
        im_resized.save(new_name)
        self.im = imageio.imread(new_name, ignoregamma = True, as_gray = True)
        self.image_size_x = len(self.im)
        self.image_size_y = len(self.im[0])

    def coord_to_freq(self, x, y):
        '''
        Converts x,y position of pixel to corresponding frequency, wraps actual
        conversion functions
        --------------------------------------------------------------------
        Inputs:
            x, y: position of pixel
        --------------------------------------------------------------------
        Outputs:
            frequency
        '''
        if self.method == "hilbert":
            return self.get_hilbert_freq(x, y)
        elif self.method == "l2r":
            return self.get_l2r_freq(x, y)

    def get_l2r_freq(self, x, y):
        '''
        Converts x,y position of pixel to corresponding frequency using l2r
        method
        --------------------------------------------------------------------
        Inputs:
            x, y: position of pixel
        --------------------------------------------------------------------
        Outputs:
            frequency
        '''
        # calculates distance
        dist = y*self.image_size_x + x
        # checks that the distance is smaller than max
        assert(dist <= self.max_dist), "x = " + str(x) + " y = " + str(y)
        # returns logarithmically spaced frequency
        return self.logspace[dist - 1]

    def get_hilbert_freq(self, x, y):
        '''
        Converts x,y position of pixel to corresponding frequency using pseudo-
        hilbert-curve
        --------------------------------------------------------------------
        Inputs:
            x, y: position of pixel
        --------------------------------------------------------------------
        Outputs:
            frequency
        '''
        #calculates distance along our hilbert curve
        dist = self.hilbert_curve.distance_from_coordinates([x, y])
        #checks that the distance is smaller than max
        assert(dist <= self.max_dist), "x = " + str(x) + " y = " + str(y)
        # returns logarithmically spaced frequency
        return self.logspace[dist - 1]

    def brightness_to_volume(self, x, y):
        '''
        Converts pixel brightness at x,y position to volume through linear mapping.
        --------------------------------------------------------------------
        Inputs:
            x, y: position of pixel
        --------------------------------------------------------------------
        Outputs:
            volume (amplitude) could consider squared mapping for intensity as I~A^2
        '''
        # maximum brightness, white pixel has this value
        max_brightness = 255

        # for white background
        if self.null_colour == "white":
            #divides pixel brightness by max brightness to get relative brightness
            #multiplies by maximum volume.
            #This ensures that brightness 0   --> volume = 0
            #                  max brightness --> max volume
            return self.im[y][x]/max_brightness * self.max_vol # USE THIS FOR 0 VOLUME = WHITE
        else:
            # for black background subtract 255 from brightness and multiply by -1
            #This ensures that white    --> volume = 0
            #                  black    --> max volume
            return -(self.im[y][x] - 255)/max_brightness * self.max_vol #use this for 0 volume = black

    def f2i(self, f):
        #maps frequency to corresponding index
        return int(self.tone_length * f)

    def complex_entry(self, mag, angle):
        # needed for fourier transform
        return mag * np.exp(1j*angle)

    def convert(self):
        '''
        Converts image to time-signal audio
        --------------------------------------------------------------------
        Inputs: (set by class)
        --------------------------------------------------------------------
        Returns:
            np.array() holding audio data at specified sample rate
        '''

        # set up initial frequency spectrum as zeros. Needs to have this length
        # to then have inverse fourier transform of desired SR, tone length
        spectrum = np.zeros(int(self.sample_rate * self.tone_length), dtype = np.csingle)

        lowest_vol = 1 #is updated later

        if self.method != "spectrogram":
        # spectrogram, not yet implemented, has to be fundamentally differently calculated

            # loop through rows
            for y, row in enumerate(self.im):
                # every new row, update progress bar in gui, not really important
                self.progress = 80 * y/self.image_size_y #80% of time taken by this

                #loop through pixels
                for x, pixel in enumerate(row):

                    # get volume
                    initial_volume = self.brightness_to_volume(x, y)
                    # get frequency
                    freq = self.coord_to_freq(x, y)
                    #set new phases
                    self.phase[freq] = 2* np.pi * freq * self.tone_length + self.phase[freq]

                    #volume is driven by how much pixel changed, at least 0.01, at most 1
                    volume = max(min(abs(self.prev_volume[freq] - initial_volume), 1), 0.01)

                    # spectrum is mostly empty, use f2i() to get correct index corresponding to right frequency
                    spectrum[self.f2i(freq)] = self.complex_entry(volume, self.phase[freq])

                    # update lowest volume
                    lowest_vol = min(volume, lowest_vol)

                    #keep track of volume changes for next iteration
                    self.prev_volume[freq] = initial_volume

        if self.force_null:
            # if we force the lowest entry to be 0 (normalise), subtract lowest volume from all
            spectrum -= lowest_vol
        # now iFFT (inverse fast fourier transform) frequency spectrum to get audio signal
        self.audio = np.real(fft.ifft(spectrum, self.tone_length * self.sample_rate))
        self.audio = self.audio/max(self.audio) #normalise audio to not be too loud/quiet

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

    def set_phase(self, phase):
        self.phase = phase

    def set_prev_vol(self, vol):
        self.prev_volume = vol

    def save_wav(self, audio, file_name):
        wavfile.write(file_name, int(self.sample_rate), audio)
        return

# OUTDATED METHOD, CALCULATE SINE WAVES MANUALLY
#    def gen_partial_sound_wave(self, initial_volume, freq, duration = 10000):
#        #duration in milliseconds
#
#        audio = []
#        one_period = []
#
#        num_samples = 1/freq * self.sample_rate
#        num_samples = self.tone_length * self.sample_rate
#        samples = np.array(range(int(num_samples)))
#
#        #if self.prev_volume[freq] == volume: return np.zeros(len(samples)) # emphasize changes
#
#        volume = max(min(abs(self.prev_volume[freq] - initial_volume), 1), 0.01) #volume is driven by how much pixel changed, at least 0.1, at most 1
#        self.prev_volume[freq] = initial_volume
#
#        one_period = volume * np.sin(2* np.pi * freq * samples/self.sample_rate + self.phase[freq])
#
#        # save phase
#        self.phase[freq] = 2* np.pi * freq * samples[-1]/self.sample_rate + self.phase[freq]
#
#        assert len(one_period) != 0
#        return one_period
#
#    def add_waves(self, waves):
#        max_period = max([len(w) for w in waves])
#        combined = np.zeros(max_period)
#        for wave in waves:
#            combined += np.resize(wave, max_period)
#
#        return combined/max(combined)



#%%
#if __name__ == "__main__":
#    #if called here access folders, for testing
#
#    waves = []
#    sample_rate = 44100.0
#    i = 0
#    max_prints = 5
#    tone_length = 5 # in seconds
#    max_freq = 18000
#
#    image_name = "testim.png"
#    image_loc = "./images/"
#    image_path = image_loc + image_name
#
#    converter = Converter(method = "hilbert", image_path = image_path, tone_length = tone_length)
#    converter.convert()
#    converter.save_wav(converter.audio, converter.audio_path)

#%%

#This converts a video, not yet wrapped in function
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

    audio = np.concatenate(audio)

    converter.save_wav(audio, "car_audio.wav")
