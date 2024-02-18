# Date:     2024-02-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define various feature extractors

import cv2
import numpy as np
from Utilities import print_execution_time

class FeatureExtractor():
    def __init__(self) -> None:
        pass

class FeatureExtractorORB(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        self._orb = cv2.ORB_create(
            edgeThreshold = 7
        )
        self.isSuccessful = False

    def ComputeFeatures(self, image) -> None:
        self._image = image

        self._image_bw = cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY) 
        self._keypoints, self._descriptors = self._orb.detectAndCompute(
            self._image_bw, None)

        # print('Number of keypoints detected:', len(self._keypoints))
        self.isSuccessful = len(self._keypoints) > 0

    def GetFeatures(self) -> tuple:
        return self._keypoints, self._descriptors
    
    def GetImage(self) -> np.ndarray:
        return self._image

    def GetKeypointsVisualization(self) -> np.ndarray:
        frameDisplay = cv2.drawKeypoints(
            self._image, self._keypoints, None, color=(255,0,0), flags=0)
        return frameDisplay


