from ImageProcess import ImageProcessor
import ImageProcess as ImP
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, cap):
        self.capturer = cap
        self.imgpr = ImageProcessor()
        self.key = -1
        self.callback = None
        self.memVar = [0, 0, 0, 0, 0]              # 0: smooth1, 1: smooth2, 2: plot, 3: rotate, 4: channel
        self.keys = ['s', 'S', 'p', 'r', 'c']
        self.trackBarProcess = [self.imgpr.changeSmoothing,
                                self.imgpr.myChangeSmoothing,
                                self.imgpr.plot,
                                self.imgpr.rotateImg,
                                self.imgpr.getImgInChannel]
        self.callbacks = [self.changeSmooth1,
                          self.changeSmooth2,
                          self.changeNPlot,
                          self.changeAngle,
                          self.changeChannel]

    def resetMemVar(self):
        self.memVar = [0,0,0,0,0]

    def display(self):
        ret, frame = self.capturer.read()
        if ret:
            self.imgpr.setFrame(cv2.flip(frame, 1))
            if self.key == -1:
                ImP.processImg(self.imgpr, ord('o'))
            elif chr(self.key) not in self.keys:
                ImP.processImg(self.imgpr, self.key)
            else:
                index = self.keys.index(chr(self.key))
                self.trackBarProcess[index](self.memVar[index])
                pass
        return ret

    def setKey(self, key):

        if chr(key) == 'w':
            ImP.processImg(self.imgpr,key)
            return

        if chr(key) == 'h':
            print (self.imgpr.displayHelp())
            return

        self.key = key
        self.resetMemVar()
        try:
            index = self.keys.index(chr(self.key))
        except ValueError:
            index = -1
        if index > -1:
            ImP.processImg(self.imgpr, self.key, self.callbacks[index])

        pass

    def changeNPlot(self, val):
        self.memVar[2] = val

    def changeSmooth1(self, val):
        self.memVar[0] = val

    def changeSmooth2(self, val):
        self.memVar[1] = val

    def changeAngle(self, val):
        self.memVar[3] = val

    def changeChannel(self, val):
        self.memVar[4] = val
