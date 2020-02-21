from converter import converter

import time
from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import filedialog
from playsound import playsound
import threading
#from tkinter.ttk import *

class App(Tk):
    def __init__(self):
        super().__init__()

        self.title("Make the visible audible")

        self.selected = IntVar()
        self.rad1 = Radiobutton(self,text='Hilbert', value=1, variable=self.selected, indicatoron = True)

        # Not functional yet
        #self.rad2 = Radiobutton(window,text='Snake', value=2, variable=selected, indicatoron = True)
        #self.rad3 = Radiobutton(window,text='Spectrogram', value=3, variable=selected, indicatoron = True)

        self.value_to_method =  {1: "hilbert",
                                 2: "snake",
                                 3: "spectrogram"}


        self.rad1.grid(column=0, row=0)
        # Not functional yet
        #self.rad2.grid(column=1, row=0)
        #self.rad3.grid(column=2, row=0)

        self.btn1 = Button(self, text="Convert Image", command=self.convert_image, state = "disabled")
        self.btn2 = Button(self, text="Upload Image", command=self.upload_image)
        self.btn3 = Button(self, text="Play Sound", command=self.play_sound, state = "disabled")
        self.btn1.grid(column=3, row=1)
        self.btn2.grid(column=3, row=0)
        self.btn3.grid(column= 3, row = 2)

        self.progress=Progressbar(self, orient=HORIZONTAL, length=100, mode='indeterminate')
        self.progress.grid(column = 3, row = 3)

        self.image_converted = False



        return

    def convert_image(self):
        self.image_converted = False
        def real_convert_image():
           self.method = self.value_to_method[self.selected.get()]
           self.audio_path = converter(self.image_path, self.method)
           self.btn1['state'] = 'normal'
           self.progress['value'] = 100
           self.image_converted = True
           self.btn3["state"] = "normal"

        def progress_bar_updater():
            while not self.image_converted:
                time.sleep(0.4)
                self.progress.step(10)

        self.btn1["state"] = "disabled"
        threading.Thread(target=real_convert_image).start()
        threading.Thread(target=progress_bar_updater).start()

        return

    def play_sound(self):
        playsound(self.audio_path)
        return

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        self.btn1["state"] = "normal"
        return

if __name__ == "__main__":
    app = App()
    app.mainloop()
