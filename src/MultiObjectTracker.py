# Date:     2024-02-08
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define a multiobjects tracker

import numpy as np
from Utilities import print_execution_time, GetNthOccurenceIndex
from scipy.spatial import distance_matrix
import BoundingBox


class MultiObjectTracker():

    def __init__(self, maxNumTrackedObjects : int) -> None:
        # Initialize parameters
        self._maxNumTrackedObjects = maxNumTrackedObjects
        self._featureSpaceSize = 10
        self._minLife = 0
        self._maxLife = 5

        # Initialize data arrays
        self._bboxes = np.zeros((self._maxNumTrackedObjects, 4))
        self._classes = np.zeros(self._maxNumTrackedObjects)
        self._features = np.zeros(
            (self._maxNumTrackedObjects, self._featureSpaceSize)
        )
        self._movementVecs = np.zeros((self._maxNumTrackedObjects, 2))
        self._trackingIDs = np.zeros(self._maxNumTrackedObjects)
        self._lifes =  np.zeros(self._maxNumTrackedObjects)

    @print_execution_time
    def Update(self, image : np.ndarray, bboxes : np.ndarray, 
               classes : np.ndarray) -> None:

        # Debug: insert arbitrary tracked objects 
        res = self.InsertNewTrackedObjects(
            np.array([[10,10,20,20]]), 64, np.ones(self._featureSpaceSize)
        )
        print(res)
        res = self.InsertNewTrackedObjects(
            np.array([[100,200,300,400]]), 64, np.ones(self._featureSpaceSize)
        )
        print(res)
        # ------

        self.PrintStatus()

        # For each unique class of detected matches
        for c in np.unique(classes):
            print('Current class is', c)

            # Between detected objects, select those which are of given class
            classSelectionMask = classes == c
            currBoxes = bboxes[classSelectionMask]

            # Between tracked object, select those which are of given class
            classSelectionMaskTrackedObj = self._classes == c
            currTrackedBoxes = self._bboxes[classSelectionMaskTrackedObj]

            # Extract features of detected matches
            currFeatures = self.ExtractFeatures(image, currBoxes)

            # Get features of tracked matches
            currTrackedFeatures = self._features[classSelectionMaskTrackedObj]
            
            # Compute predicted position of tracked objects
            currMovementVecs = self._movementVecs[classSelectionMaskTrackedObj]
            currPredictedTrackedBoxes = self.GetPredictedPosition(
                currTrackedBoxes, currMovementVecs)
            #print(currPredictedTrackedBoxes)

            # Extract matrix of distances between each detected match of given 
            # class and each prediction of tracked match of given class
            # Also extract matrix of feature distances between each match
            nsize = currBoxes.shape[0]
            centersDistancesMatrix = np.ones((nsize,nsize)) * 1e9
            featuresDistancesMatrix = np.ones((nsize,nsize)) * 1e9
            if(currPredictedTrackedBoxes.shape[0] > 0):
                detCenters = np.apply_along_axis(
                    BoundingBox.GetCenter, 1, currBoxes
                )
                trackedCenters = np.apply_along_axis(
                    BoundingBox.GetCenter, 1, currPredictedTrackedBoxes
                )
                centersDistancesMatrix = self.ComputeDistancesMatrix(
                    detCenters, trackedCenters
                )
                featuresDistancesMatrix = self.ComputeDistancesMatrix(
                    currFeatures, currTrackedFeatures
                )
            #distancesMatrixTEST = self.ComputeDistancesMatrix(currBoxes, currBoxes)
            print(centersDistancesMatrix)
            print(featuresDistancesMatrix)

            # Based of center distances and features distances, classify each
            # match as either:
            # - correspondent (get corresponding tracked object index)
            # - occlusion (get corresponding tracked object index)
            # - new match
            # TODO
            
            
            # Then, with respect to all tracked objects, perform action
            # - for those which are correspondent, compute movement vector, 
            #   update box, features and add life integer by one, up to max
            # - for those which are new matches, update box, features, movement 
            #   vector as zeros and add life integer by one, up to max
            # - for those which are in occlusion, propagate box by movement vector 
            #   but do not change life integer
            # TODO


        # Finally, for all others, propagate by movement vector and also decrease 
        # life integer by one, down to life min
        # TODO   



    def ExtractFeatures(self, image : np.ndarray, bboxes : np.ndarray) -> np.ndarray:
        # Dummy feature extraction
        shape = bboxes.shape
        print(shape)
        return np.ones((shape[0],self._featureSpaceSize))
  
    def GetPredictedPosition(self, boxes : np.ndarray, 
                             movementVec : np.ndarray) -> np.ndarray:
        # Dummy predicted position
        #print(boxes)
        #print(boxes.shape)
        #print(movementVec)
        #print(movementVec.shape)
        return boxes

    def ComputeDistancesMatrix(self, centersA : np.ndarray, centersB : np.ndarray):
        distancesMatrix = distance_matrix(centersA, centersB)
        return distancesMatrix


    def InsertNewTrackedObjects(self, box : np.ndarray, label : int, 
                                features : np.ndarray) -> bool:
        # Check if space is present for inserting a tracking object
        maskAvailable = np.logical_and(self._classes == 0, self._lifes == 0)
        print(maskAvailable)
        currIndex = GetNthOccurenceIndex(maskAvailable, 0)
        print(currIndex)

        if(currIndex >= 0):
            # Space is available for tracking one more object
            self._bboxes[currIndex] = box
            self._classes[currIndex] = label
            self._features[currIndex] = features
            self._movementVecs[currIndex] = np.zeros(2)
            self._trackingIDs[currIndex] = self.GetNextUniqueTrackID()
            self._lifes[currIndex] = self._minLife + 1
            return True
        else:
            # No space is left for tracking object
            return False

    def GetNextUniqueTrackID(self) -> int:
        return np.max(self._trackingIDs) + 1

    def PrintStatus(self) -> None:
        print('== TRACKED OBJECTS ==')
        print('Boxes:')
        print(self._bboxes)
        print('Classes:')
        print(self._classes)
        print('Features:')
        print(self._features)
        print('Movement vectors:')
        print(self._movementVecs)
        print('Tracking IDs:')
        print(self._trackingIDs)
        print('Lives:')
        print(self._lifes)
        print('===================')

    def UpdatePositionPrediction(self):
        pass

    def GetFeaturesDistance(self):
        pass
    
