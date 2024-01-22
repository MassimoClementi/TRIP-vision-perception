# Date:     2023-01-31
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Class that defines a custom framegrabber

import cv2

class Framegrabber:
    stream_ended = False
    __scaling_factor = 1.

    def __init__(self, path):
        """Instantiate an input manager"""
        print('Opening framegrabber...')
        self.path = path
        self.cap = cv2.VideoCapture(path)
        assert self.check_cap()

    def check_cap(self):
        """Check that the video capture is open"""
        return self.cap.isOpened()

    def is_ended(self):
        """Return if the stream has ended (no frames left)"""
        return self.stream_ended

    def grab_frame(self):
        """Get the current frame"""

        # Capture frame-by-frame
        print('Grabbing frame...')
        ret, frame = self.cap.read()

        # check if frame is read correctly
        if not ret:
            self.stream_ended = True
            return

        # Resize frame image
        frame_resized = cv2.resize(frame,
                                   (int(frame.shape[1] * self.__scaling_factor),
                                    int(frame.shape[0] * self.__scaling_factor)),
                                   interpolation=cv2.INTER_AREA)
        
        return frame_resized

    def cap_release(self):
        print('Closing framegrabber...')
        return self.cap.release()

    def set_scaling_factor(self, scaling_factor):
        self.__scaling_factor = scaling_factor
