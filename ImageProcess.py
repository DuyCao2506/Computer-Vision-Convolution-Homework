# coding=utf-8
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
import scipy
import argparse
from enum import Enum


def draw_arrow(image, x, y, dx, dy, color, arrow_magnitude=9, thickness=1, line_type=4, shift=0):
    # calc angle of the arrow
    angle = np.arctan2(dx, dy)
    x1 = int(arrow_magnitude * np.cos(angle) + x)
    y1 = int(arrow_magnitude * np.sin(angle) + y)
    p = (int(y), int(x))
    q = (y1, x1)

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)

    # # starting point of first line of arrow head
    # p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    # int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # # draw first half of arrow head
    # cv2.line(image, p, q, color, thickness, line_type, shift)
    # # starting point of second line of arrow head
    # p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    # int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # # draw second half of arrow head
    # cv2.line(image, p, q, color, thickness, line_type, shift)

class Tool(Enum):
    SMOOTH = 1
    SMOOTH2 = 2
    ANGLE = 3
    PATH = 4



class ImageProcessor:
    "Image Processing Instances"
    def __init__(self, imagePath=None, wdname="Frame"):
        self.path = imagePath

        if imagePath != None:
            self.img = self.reloadImg()
            self.cloneImg = self.reloadImg()
            self.height = len(self.img)
            self.width = len(self.img[0])
        self.channel = 0
        self.smoothVal = 0
        self.tools = []
        self.windowName = wdname

    def setFrame(self, img):
        self.img = img
        self.cloneImg = np.zeros((img.shape),dtype="uint8")
        self.cloneImg[:, :, :] = self.img[:, :, :]

    def showImg(self):
        cv2.imshow(self.windowName,self.cloneImg)

    def displayTool(self, toolname, callback):
        if toolname == Tool.SMOOTH:
            cv2.createTrackbar('Smooth', self.windowName, 0, 15, callback)
        elif toolname == Tool.SMOOTH2:
            cv2.createTrackbar('Smooth Bar', self.windowName, 0, 15, callback)
        elif toolname == Tool.ANGLE:
            cv2.createTrackbar('Angle',self.windowName, 0, 180, callback)
        elif toolname == Tool.PATH:
            cv2.createTrackbar('Distance', self.windowName, 0, 30, callback)

    def reloadImg(self):
        img = cv2.imread(self.path, 1)
        return img

    def save(self):
        cv2.imwrite('output.jpg',self.cloneImg)
        pass

    #####################
    # built-in functions
    #####################

    def cvGrayScale(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        return img_gray

    def changeSmoothing(self, value):
        img = self.cvGrayScale()
        i = value*2 + 1
        self.cloneImg = cv2.GaussianBlur(img,(i, i),0)
        cv2.imshow(self.windowName,self.cloneImg)



    #####################
    # user-defined functionsG
    #####################
    def plot(self, N):
        N+=1
        K = 20
        dx = self.convX()
        dy = self.convY()
        xlength, ylength = self.cloneImg.shape
        for x in np.arange(1, xlength, N*5):
            for y in np.arange(1, ylength, N*5):
                draw_arrow(self.cloneImg, x, y, dx[x][y], dy[x][y], [100, 0, 255], K)

        cv2.imshow(self.windowName, self.cloneImg)


    def rotateImg(self, val):
        self.cloneImg = self.cvGrayScale(self.img)
        rows, cols = self.cloneImg.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), val, 1)
        self.cloneImg = cv2.warpAffine(self.cloneImg, M, (cols, rows))
        cv2.imshow(self.windowName, self.cloneImg)

    def normalize(self, beta, alpha, img):
        mins = np.ones((img.shape[0], img.shape[1])) * img.min()
        maxs = np.ones((img.shape[0], img.shape[1])) * img.max()
        return (img - mins)*(beta - alpha)/(maxs - mins) + alpha

    def magnitudeGradient(self):
        self.cloneImg = self.cvGrayScale()
        sobx_p2 = cv2.Sobel(self.cloneImg,cv2.CV_64F, 1, 0, 3)
        sobx_p2 **= 2
        soby_p2 = cv2.Sobel(self.cloneImg,cv2.CV_64F, 0, 1, 3)
        soby_p2 **= 2
        return np.asanyarray(self.normalize(255, 0, np.sqrt(sobx_p2 + soby_p2)),dtype="uint8")

    def myGrayScale(self):
        r = self.img[:, :, 0]
        g = self.img[:, :, 1]
        b = self.img[:, :, 2]
        dest = (0.299*r + 0.587*g + 0.114*b)
        dest = np.asanyarray(dest, dtype=np.uint8)
        return dest

    def average(self,pxl):
        return np.average(pxl)


    def getImgInChannel(self, channel):
        self.cloneImg = np.zeros(self.img.shape,"uint8")
        self.cloneImg[:, :, channel] = self.img[:, :, channel]
        cv2.imshow(self.windowName, self.cloneImg)
        return self.cloneImg

    def myChangeSmoothing(self, value):
        value = 2*value+1
        self.cloneImg = self.cvGrayScale()
        tmp = self.cvGrayScale()
        kernel = self.getKernel(value)
        self.cloneImg = self.convolve(self.cloneImg, kernel)

        # for x in range(self.height):
        #     for y in range(self.width):
        #         corMat = self.getCoreMat(value, tmp, x, y)
        #         colves = self.eleConvolve(kernel, corMat, value)
        #         self.cloneImg[x][y] = colves

        cv2.imshow(self.windowName, self.cloneImg)
        pass

    def getKernel(self, value):
        kernel = (1.0/(value*value)) * np.ones((value, value), dtype=np.float32)
        return kernel
        pass

    def convX(self):
        self.cloneImg = self.cvGrayScale()
        img = cv2.Sobel(self.cloneImg, cv2.CV_64F, 0, 1, 3)
        return np.asanyarray(self.normalize(255, 0, img), dtype=np.uint8)


    def convY(self):
        self.cloneImg = self.cvGrayScale()
        img = cv2.Sobel(self.cloneImg, cv2.CV_64F, 1, 0, 3)
        return np.asanyarray(self.normalize(255, 0, img), dtype=np.uint8)



    def displayHelp(self):
        strin = "i - reload the original image (i.e. cancel any previous processing) \n" \
                "w - save the current (possibly processed) image into the file out.jpg  \n " \
                "g - convert the image to grayscale using the openCV conversion function.  \n" \
                "G - convert the image to grayscale using your implementation of conversion function.  \n" \
                "c - cycle through the color channels of the image showing a different channel every time the key is pressed. \n " \
                "s - convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.   \n" \
                "S - convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing.  \n" \
                "x - convert the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255].  y - convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255]. \n" \
                "m - show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image.   \n" \
                "p - convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K. \n" \
                "r - convert the image to grayscale and rotate it using an angle of Q degrees. Use a track bar to control the rotation angle. The rotation of the image should be performed using an inverse map so there are no holes in it.   \n" \
                "h - Display a short description of the program, its command line arguments, and the keys it supports.   \n"
        return strin

    def convolve(self, image, kernel):
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]

        pad = (kW - 1) / 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):

                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                k = (roi * kernel).sum()

                output[y - pad, x - pad] = k
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        # return the output image
        return output


###############
# main-function
# ###############


def processImg(ins, key, callback=None):
    if chr(key) == 'i':
        ins.cloneImg = ins.reloadImg()
    elif chr(key) == 'w':
        ins.save()
    elif chr(key) == 'g':
        ins.cloneImg = ins.cvGrayScale()
    elif chr(key) == 'G':
        ins.cloneImg = ins.myGrayScale()
    elif chr(key) == 'c':
        ins.cloneImg = ins.getImgInChannel(ins.channel)
        ins.channel = (ins.channel + 1) % 3
        callback(ins.channel) if  not (callback is None) else None
    elif chr(key) == 's':
        ins.tools.append(Tool.SMOOTH)
        ins.displayTool(Tool.SMOOTH, ins.changeSmoothing if callback is None else callback)
    elif chr(key) == 'S':
        ins.tools.append(Tool.SMOOTH2)
        ins.displayTool(Tool.SMOOTH2, ins.myChangeSmoothing if callback is None else callback)
        return
    elif chr(key) == 'x':
        ins.cloneImg = ins.convX()
    elif chr(key) == 'y':
        ins.cloneImg =ins.convY()
    elif chr(key) == 'm':
        ins.cloneImg = ins.magnitudeGradient()
    elif chr(key) == 'p':
        ins.tools.append(Tool.PATH)
        ins.displayTool(Tool.PATH, ins.plot if callback == None else callback)
        return
    elif chr(key) == 'r':
        ins.tools.append(Tool.ANGLE)
        ins.displayTool(Tool.ANGLE, ins.rotateImg if callback == None else callback)
        return
    elif chr(key) == 'h':
        print (ins.displayHelp())
        return
    else:
        pass

    cv2.imshow(ins.windowName, ins.cloneImg)
    pass

