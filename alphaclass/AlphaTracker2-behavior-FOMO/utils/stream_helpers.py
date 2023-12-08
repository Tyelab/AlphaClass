import cv2
import numpy as np
import os
import json

from threading import Thread

import numpy as np
import os
import cv2
from tqdm.notebook import tqdm
import time

import imutils
from imutils.video import WebcamVideoStream

from utils.image_helpers import letterbox



class Streams(WebcamVideoStream):
    def __init__(self, src, buffer=True):
        super().__init__(src)

        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.buffer = buffer

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()
            if self.buffer:
                time.sleep(1/self.fps)

    def stop(self):
        cv2.destroyAllWindows()
        self.stopped=True
        self.stream.release()




class StreamViewer:
    def __init__(self, sources, size=(320, 256), buffer=True, letterbox='yes'):

        assert type(sources) == list
        self.sources = sources
        self.size = size
        self.buffer = buffer
        self.num_sources = len(self.sources)

        self.captured = []
        for s_count, s in enumerate(self.sources):
            #wc = WebcamVideoStream(s)
            wc = Streams(s, self.buffer)
            wc.start()
            if wc.grabbed:
                print(f'capture from {s}: success')
                self.captured.append(wc)
            else:
                print(f'capture from {s} failed')
                self.captured.append(None)


    def stop(self):
        for s in self.captured:
            s.stop()


    def update(self):
        self.img_stack = [i.read() for i in self.captured]
        none_check = any(elem is None for elem in self.img_stack)
        where_none = [n for n, n_ in enumerate(self.img_stack) if n_ is None]
        if where_none:
            for stop in where_none:
                self.captured[stop].stop()

        else:
            self.img_stack = np.array(self.img_stack)
            if letterbox=='yes':
                self.img_stack = [letterbox(i, (self.size[1], self.size[0]))[0] for i in self.img_stack]

            return self.img_stack
