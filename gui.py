from converter import Converter

import time
from tkinter.ttk import Progressbar
from tkinter import filedialog
from playsound import playsound
import numpy as np
import threading
import pygame
import pygame.camera
from tkinter import *
pygame.init()
pygame.camera.init()
import sounddevice as sd
#from tkinter.ttk import *
import queue

class App(Tk):

    ''' Graphical User Interface for taking/uploading images to convert'''

    def __init__(self):
        super().__init__() #Use Tk

        self.title("Make the visible audible")

        # Initialise webcam
        pygame.camera.init()
        camlist = pygame.camera.list_cameras()
        self.cam = pygame.camera.Camera(camlist[0], (16, 16))

        # Tone duration
        self.tone_length = 5 #sec

        # All buttons
        self.selected = IntVar()
        self.rad1 = Radiobutton(self, text='Hilbert', value=1, variable=self.selected, indicatoron = True,
                                    command = self.pressed_radios)
        self.rad2 = Radiobutton(self, text='Snake', value=2, variable=self.selected, indicatoron = True,
                                    command = self.pressed_radios)

        # Not functional yet
        #self.rad3 = Radiobutton(self ,text='Spectrogram', value=3, variable=selected, indicatoron = True)

        self.value_to_method =  {1: "hilbert",
                                 2: "snake",
                                 3: "spectrogram"}


        self.rad1.grid(column=0, row = 1)
        self.rad2.grid(column=1, row = 1)

        #self.radio_title = Label(self, text="CONVERSION METHOD:", font=("system", 10), compound = CENTER)
        #self.radio_title.grid(column = 0, row = 0, columnspan = 2)


        # Not functional yet
        #self.rad3.grid(column=2, row = 1)

        normal_button_width = 12
        normal_button_height = 2

        #self.btn1 = Button(self, text="Convert Image", command=self.convert_image, state = "disabled",
                            #height = normal_button_height, width = 2* normal_button_width)

        self.btn2 = Button(self, text="Upload Image", command=self.upload_image,
                            height = normal_button_height, width = normal_button_width, state = "disabled")

        self.btn3 = Button(self, text="Play Sound", command=self.play_sound, state = "disabled")

        self.btn_takeim = Button(self, text="Take Picture", command=self.take_image,
                            height = normal_button_height, width = normal_button_width, state = "disabled")
        self.btn_begin_cap = Button(self, text="Begin Capture", command=self.begin_capture, state = "disabled",
                            height = normal_button_height, width = normal_button_width)
        self.btn_end_cap = Button(self, text="Stop Capture", command=self.end_capture, state = "disabled",
                            height = normal_button_height, width = normal_button_width)

        #self.btn1.grid(column=3, row=1, columnspan = 2, sticky="nesw")
        self.btn2.grid(column=3, row = 1)
        self.btn3.grid(column= 3, row = 2, sticky="nesw", columnspan = 2)

        self.btn_begin_cap.grid(column = 5, row = 1, sticky = "nesw")
        self.btn_end_cap.grid(column = 5, row = 2)
        self.btn_takeim.grid(column = 4, row = 1)

        # Progress bar for converting images
        self.progress=Progressbar(self, orient=HORIZONTAL, length=150, mode='determinate')
        self.progress.grid(column = 0, row = 2, padx = 10, columnspan = 2)


        self.image_converted = False

        #colours

        self.green = "#b6ffad"
        self.red   = "#FF9173"

        # Seperators
        #self.sep1 = ttk.Separator(self, orient=VERTICAL)
        #self.sep2 = ttk.Separator(self, orient=VERTICAL)
#
        #self.sep1.grid(column=2, row=0, rowspan=3, sticky='ns', padx=3, pady = 3)
        #self.sep2.grid(column=4, row=0, rowspan=1, sticky='ns', padx=3, pady = 3)


        return

    def take_image(self):
        #self.btn1["state"] = "normal"
        #self.btn1.configure(bg = self.green)
        self.cam.start()
        im = self.cam.get_image()
        im = pygame.transform.scale(im, (16, 16)) #make 16x16
        im_array = pygame.surfarray.array3d(im)
        im_array = im_array.dot([0.298, 0.587, 0.114])[:,:,None] #make black and white
        name = time.strftime("%H-%M-%S", time.localtime())
        pygame.image.save(im, "./images/" + name + ".png")
        self.image_path = "./images/" + name + ".png"
        self.cam.stop()
        self.tone_length = 5
        self.convert_image()
        return

    def pressed_radios(self):
        self.btn2["state"] = "normal"
        self.btn_takeim["state"] = "normal"
        self.btn_begin_cap["state"] = "normal"

        self.btn2.config(bg = self.green)
        self.btn_begin_cap.config(bg = self.green)
        self.btn_takeim.config(bg = self.green)

    def convert_image(self, im = None):
        self.image_converted = False
        def real_convert_image():
            self.method = self.value_to_method[self.selected.get()]
            self.converter = Converter(self.method, self.image_path, tone_length = self.tone_length)
            self.converter.set_null_colour = "white" #white background
            #self.converter.force_null = True
            self.audio = self.converter.convert()
            self.audio_path = self.converter.save_audio()
            #self.btn1['state'] = 'normal'
            self.image_converted = True
            self.btn3["state"] = "normal"
            self.btn3.configure(bg = self.green)

        def progress_bar_updater():
            while not self.image_converted:
                time.sleep(0.3)
                self.progress["value"] = self.converter.progress
            self.progress["value"] = 100

        #self.btn1["state"] = "disabled"
        self.btn3["state"] = "disabled"
        threading.Thread(target=real_convert_image).start()
        threading.Thread(target=progress_bar_updater).start()

        return

    def play_sound(self):
        #playsound(self.audio_path)
        sd.play(list(self.audio), 44100)
        return

    def begin_capture(self):
        self.tone_length = 1 #shorter tones for stream
        self.btn_end_cap["state"] = "normal"
        self.btn_begin_cap["state"] = "disabled"
        self.btn2["state"] = "disabled"
        self.btn2.configure(bg = "lightgrey")
        self.btn_takeim.configure(bg = "lightgrey")
        self.btn_takeim["state"] = "disabled"
        self.btn_begin_cap.configure(bg = "lightgrey")
        self.btn_end_cap.configure(bg = self.red)
        method = self.value_to_method[self.selected.get()]
        self.capture = Capture()
        self.audio_stream = AudioStream(method=method)
        self.capture.start()
        self.audio_stream.start()

    def end_capture(self):
        self.capture.going = False
        self.audio_stream.stream.stop() #not while loop, harder to stop like capture
        self.btn_end_cap["state"] = "disabled"
        self.btn_begin_cap["state"] = "normal"
        self.btn2["state"] = "normal"
        self.btn2.configure(bg = self.green)
        self.btn_takeim.configure(bg = self.green)
        self.btn_takeim["state"] = "normal"
        self.btn_end_cap.configure(bg = "lightgrey")
        self.btn_begin_cap.configure(bg = self.green)

#        def retrieve_im():
#            self.is_recording = True
#            print("this is executed")
#            while self.is_recording:
#                time.sleep(500)
#                print("getting here")
#                if not q.empty():
#                    im_array = q.get()
#                    self.convert_image(im = im_array)
#                    print(im_array)
#                    self.play_sound()

        #capture_thread = threading.Thread(target=capture.main())
        #retrieve_thread = threading.Thread(target=retrieve_im())
        #capture_thread.start()
        #retrieve_thread.start()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        #self.btn1["state"] = "normal"
        #self.btn1.configure(bg = self.green)
        self.convert_image()
        return

class Capture(threading.Thread):
    def __init__(self):

        self.j = 0
        threading.Thread.__init__(self)
        self.daemon = True
        self.size = (160,120) #minimum by camera
        #self.size = (16, 16)
        # create a display surface. standard pygame stuff
        self.display = pygame.display.set_mode(self.size, 0)

        # this is the same as what we saw before
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")
        self.cam = pygame.camera.Camera(self.clist[0], self.size)
        self.cam.start()

        # create a surface to capture to.  for performance purposes
        # bit depth is the same as that of the display surface.
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_and_flip(self):
        # if you don't want to tie the framerate to the camera, you can check
        # if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        #print("cam size: ", self.cam.get_size())
        if self.cam.query_image():
            self.j += 1
            #print(self.j)
            #print(self.snapshot.get_size())
            #print(self.cam.get_size())
            im = self.cam.get_image(self.snapshot)
            #im = self.cam.get_image()
            im = pygame.transform.scale(im, (16, 16)) #make 16x16
            im_array = pygame.surfarray.array3d(im)

            # blit it to the display surface.  simple!
            self.display.blit(self.snapshot, (0,0))
            pygame.display.flip()


            self.im_array = im_array.dot([0.298, 0.587, 0.114])[:,:,None] #make black and white

            if not im_q.full():
                im_q.put(self.im_array)
        #else:
            #print("CAMERA NOT TAKING PICTURES QUICKLY ENOUGH, REDUCE FRAMERATE")

        #name = time.strftime("%H-%M-%S", time.localtime())
        #pygame.image.save(im, "./images/" + name + ".png")
        #self.image_path = "./images/" + name + ".png"
        #self.cam.stop()
        #self.convert_image()

    def run(self):
        self.going = True
        while self.going:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    # close the camera safely
                    self.cam.stop()
                    going = False
                    print("EXITED SAFELY")
            self.get_and_flip()
        self.cam.stop()

#class AudioStream(threading.Thread):
#    def __init__(self):
#        threading.Thread.__init__(self)
#        self.daemon = True
#        self.size = (16,16)
#        self.running = True
#
#    def play_sound(self):
#
#        return
#
#    def run(self):
#        #method = app.value_to_method[self.selected.get()]
#        method = "hilbert"
#        #self.converter.force_null = True
#        while self.running:
#            time.sleep(5)
#            if not im_q.empty():
#                print("NR ITEMS IN QUEUE: ", im_q.qsize())
#                im = im_q.get()
#                converter = Converter(method, None, im = im)
#                converter.set_null_colour = "white" #white background
#                self.audio = converter.convert()
#                self.play_sound()

class AudioStream(threading.Thread):
    def __init__(self, method):
        threading.Thread.__init__(self)
        self.daemon = True

        self.running = True
        self.method = method
        self.audio = np.zeros((int(SAMPLERATE * FRAME_PERIOD), 1))


        #method = app.value_to_method[self.selected.get()]

    def callback(self, outdata, frames, time, status = False):
        #print(im_q.qsize())
        #if status:
            #print(status)

        #print('time info: ', time)
        previous_phases = None

        if not im_q.empty():
            im = im_q.get()
            converter = Converter(self.method, None, im = im)
            if previous_phases:
                converter.set_phase(previous_phases)
            converter.set_null_colour = "white" #white background
            self.audio = converter.convert()
            self.audio = np.reshape(self.audio, (-1, 1))
            if self.audio.shape[0] > frames:
                self.audio = self.audio[0:frames]

            previous_phases = converter.phase
        else:
            print("QUEUE IS EMPTY, REDUCE FRAMERATE")
        outdata[:] = self.audio
        return

    def run(self):
        with sd.OutputStream(channels = 1, blocksize = int(SAMPLERATE * FRAME_PERIOD),
                            callback = self.callback, samplerate = SAMPLERATE) as self.stream:

            sd.sleep(int(SAMPLERATE*FRAME_PERIOD*1000)) #ms
        return

if __name__ == "__main__":
    BUF_SIZE = 1
    FRAMERATE = 7
    SAMPLERATE = 44100
    FRAME_PERIOD = 1/FRAMERATE
    im_q = queue.Queue(BUF_SIZE)
    #capture = Capture()
    #capture.start()
    #audio_stream = AudioStream()

    #audio_stream.start()
    #i = 0
    #while True:
        #time.sleep(1)
        #print("t = ", i)
        #i+=1
    app = App()
    app.mainloop()

#app = App()

#capture = Capture()
