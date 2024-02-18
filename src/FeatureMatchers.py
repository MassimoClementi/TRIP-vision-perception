# Date:     2024-02-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define feature matchers

import cv2
import numpy as np
from FeatureExtractors import FeatureExtractorORB
from Utilities import print_execution_time


class FeatureMatcher():
    def __init__(self) -> None:
        pass

class FeatureMatcherORB(FeatureMatcher):
    def __init__(self, numFeatures = 25) -> None:
        super().__init__()
        self._numFeatures = numFeatures
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def ComputeMatchingFeatures(self, featureExtractor1 : FeatureExtractorORB,
                                featureExtractor2 : FeatureExtractorORB):
        
        self._featureExtractor1 = featureExtractor1
        self._featureExtractor2 = featureExtractor2

        _, descriptors1 = self._featureExtractor1.GetFeatures()
        _, descriptors2 = self._featureExtractor2.GetFeatures()

        if not self._AssertFeatureExtractionSuccessful():
            return

        matches = self._bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key = lambda x:x.distance)
        # print('Number of detected matches:', len(matches))

        self._selectedMatches = matches[:self._numFeatures]

        # for m in self._selectedMatches:
        #     print(m.distance)

    def _AssertFeatureExtractionSuccessful(self) -> bool:
        return self._featureExtractor1.isSuccessful and \
                self._featureExtractor2.isSuccessful

    def GetMatchingLoss(self) -> float:
        '''Lower is better'''
        if not self._AssertFeatureExtractionSuccessful():
            return 1e9
        distances = [d.distance for d in self._selectedMatches]
        return np.mean(distances)

    def GetMatchesVisualization(self) -> np.ndarray:
        frameDisplay = cv2.drawMatches(
            self._featureExtractor1.GetImage(),
            self._featureExtractor1.GetFeatures()[0],
            self._featureExtractor2.GetImage(),
            self._featureExtractor2.GetFeatures()[0],
            self._selectedMatches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return frameDisplay