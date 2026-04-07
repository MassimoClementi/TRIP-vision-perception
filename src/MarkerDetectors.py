# Date:     2024-09-14
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Class that defines structures to manage Aruco markers

import numpy as np
import cv2


class MarkerDetector():
    def __init__(self) -> None:
        pass


class ArucoDetectionResults():
    corners = []
    ids = []
    rejectedMatches = []

    def __init__(self) -> None:
        pass

    def Set(self, corners, ids, rejectedMatches) -> None:
        self.corners = corners
        self.ids = ids
        self.rejectedMatches = rejectedMatches

    def Get(self) -> tuple:
        return self.corners, self.ids, self.rejectedPoints

    def GetNumberMarkersFound(self) -> int:
        if self.corners == None or len(self.corners) == 0:
            return 0
        return len(self.corners)
        
    def GetNumberMarkersRejected(self) -> int:
        if self.rejectedMatches == None or len(self.rejectedMatches) == 0:
            return 0
        return len(self.rejectedMatches)

    def GetResultsOverlay(self, image : np.ndarray) -> np.ndarray:
        outputImage = image.copy()

        for c in self.corners:
            baricenter = np.average(c, axis=1).squeeze().astype(int)
            cv2.circle(outputImage, baricenter, 6, (0,255,0), -1)

        for r in self.rejectedMatches:
            baricenter = np.average(r, axis=1).squeeze().astype(int)
            cv2.circle(outputImage, baricenter, 3, (0,0,255), -1)

        return outputImage


class ArucoMarkersDetector(MarkerDetector):
    def __init__(self, dictionaryInt : int) -> None:
        super().__init__()
        self._arucoDictionary = cv2.aruco.getPredefinedDictionary(dictionaryInt)
        self._detector = self._GetDetector()

    def _GetDetector(self) -> cv2.aruco.ArucoDetector:
        detectorParameters = cv2.aruco.DetectorParameters()
        refineParameters = cv2.aruco.RefineParameters()
        return cv2.aruco.ArucoDetector(
            self._arucoDictionary, 
            detectorParameters,
            refineParameters
        )
    
    def DetectMarkers(self, image :np.ndarray) -> ArucoDetectionResults:
        corners, ids, rejectedMatches = self._detector.detectMarkers(image)
        results = ArucoDetectionResults()
        results.Set(corners, ids, rejectedMatches)
        print('Markers found: ', results.GetNumberMarkersFound())
        print('Rejected points', results.GetNumberMarkersRejected())
        return results


