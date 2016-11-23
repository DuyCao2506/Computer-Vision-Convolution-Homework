###
#Cao Khac Le Duy
#1351008
###

import cv2
import numpy as np
import sys
import ImageProcess as ImP
import VideoProcess as VidP


def captureEnv():
    cap = cv2.VideoCapture(0)
    processor = VidP.VideoProcessor(cap)
    filter = None


    while (True):

        # Capture frame-by-frame
        if processor.display():
            # Display the resulting frame
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key != -1:
                processor.setKey(key)
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass


def imageLoadEnv(argv):
    path = argv[0]

    processor = ImP.ImageProcessor(path, 'Image Processing')
    processor.showImg()
    key = cv2.waitKey(0)

    while key != 27:
        ImP.processImg(processor, key)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    pass


def main(argv):
    if len(argv) == 0:
        captureEnv()
        return
    elif len(argv) == 1:
        imageLoadEnv(argv)
    else:
        print ("Wrong arguments")

    pass

if __name__ == "__main__":
    main(sys.argv[1:])
