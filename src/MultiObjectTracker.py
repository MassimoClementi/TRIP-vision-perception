# Date:     2024-02-08
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define a multiobjects tracker

import numpy as np
from Utilities import print_execution_time, GetNthOccurenceIndex, \
                        GetIndexesFromMask
from scipy.spatial import distance_matrix
from Enums import MatchClassification
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
        self._labels = np.zeros(self._maxNumTrackedObjects, dtype=int)
        self._features = np.zeros(
            (self._maxNumTrackedObjects, self._featureSpaceSize)
        )
        self._movementVecs = np.zeros((self._maxNumTrackedObjects, 2))
        self._trackingIDs = np.zeros(self._maxNumTrackedObjects, dtype=int)
        self._lives =  np.zeros(self._maxNumTrackedObjects, dtype=int)

    @print_execution_time
    def Update(self, image : np.ndarray, dBoxes : np.ndarray, 
               dLabels : np.ndarray) -> None:

        currtUpdated =  np.zeros(self._maxNumTrackedObjects, dtype=bool)

        # For each unique class of detected matches
        for l in np.unique(dLabels):
            print('Current label is', l)

            # Between detected objects, select those which are of given class
            dCurrLabelMask = dLabels == l
            dCurrLabelIndexes = GetIndexesFromMask(dCurrLabelMask)
            dBoxesCurrLabel = dBoxes[dCurrLabelMask]

            # Between tracked object, select those which are of given class
            tCurrLabelMask = np.logical_and(self._labels == l, self._lives > 0)
            tCurrLabelIndexes = GetIndexesFromMask(tCurrLabelMask)
            tBoxesCurrLabel = self._bboxes[tCurrLabelMask]

            # Extract features of detected matches
            dFeaturesCurrLabel = self.ExtractFeatures(image, dBoxesCurrLabel)

            # Get features of tracked matches
            tFeaturesCurrLabel = self._features[tCurrLabelMask]
            
            # Compute predicted position of tracked objects
            tMovementVecsCurrLabel = self._movementVecs[tCurrLabelMask]
            tPredictedBoxesCurrLabel = self.GetPredictedPosition(
                tBoxesCurrLabel, tMovementVecsCurrLabel)

            # Extract matrix of distances between each detected match of given 
            # class and each prediction of tracked match of given class
            # Also extract matrix of feature distances between each match
            nsize = dBoxesCurrLabel.shape[0]
            matrixCentersDistances = np.ones((nsize,nsize)) * 1e9
            matrixFeaturesDistances = np.ones((nsize,nsize)) * 1e9
            if(tPredictedBoxesCurrLabel.shape[0] > 0):
                dCentersCurrLabel = np.apply_along_axis(
                    BoundingBox.GetCenter, 1, dBoxesCurrLabel
                )
                tPredictedCentersCurrLabel = np.apply_along_axis(
                    BoundingBox.GetCenter, 1, tPredictedBoxesCurrLabel
                )
                matrixCentersDistances = self.ComputeDistancesMatrix(
                    dCentersCurrLabel, tPredictedCentersCurrLabel
                )
                matrixFeaturesDistances = self.ComputeDistancesMatrix(
                    dFeaturesCurrLabel, tFeaturesCurrLabel
                )

            # Based of center distances and features distances, classify each
            # match as either:
            # - correspondent (get corresponding tracked object index)
            # - occlusion (get corresponding tracked object index)
            # - new match
            weightFactor = 1.0
            matrixWeightedDistances = weightFactor * matrixCentersDistances + \
                (1.0 - weightFactor) * matrixFeaturesDistances
            dMatchesClassification = np.apply_along_axis(
                    self.GetDetectedMatchClassification, 1, matrixWeightedDistances
            )
            print(dMatchesClassification)
            

            # Then, with respect to all tracked objects, perform action
            # - for those which are correspondent, compute movement vector, 
            #   update box, features and add life integer by one, up to max
            # - for those which are new matches, update box, features, movement 
            #   vector as zeros and add life integer by one, up to max
            # - for those which are in occlusion, propagate box by movement vector 
            #   but do not change life integer
            for dIdx in range(dMatchesClassification.shape[0]):
                currClassRes = dMatchesClassification[dIdx][0]
                currCorrespondIndex = dMatchesClassification[dIdx][1]
                
                if currClassRes == MatchClassification.CORRESPONDENT:
                    currtIndex = tCurrLabelIndexes[currCorrespondIndex]
                    print("Updating tracked object at index", currtIndex, "...")
                    currtUpdated[currtIndex] = True
                    self._movementVecs[currtIndex] = BoundingBox.GetDelta(
                        dBoxesCurrLabel[dIdx], self._bboxes[currtIndex]
                    )
                    self._bboxes[currtIndex] = dBoxesCurrLabel[dIdx]
                    self._lives[currtIndex] =+ self.UpdateLifeIndex(
                        self._lives[currtIndex], True
                    )
                if currClassRes == MatchClassification.OCCLUSION:
                    currtIndex = tCurrLabelIndexes[currCorrespondIndex]
                    print("Tracked object at index", currtIndex, "in occlusion")
                    currtUpdated[currtIndex] = True
                    pass
                if currClassRes == MatchClassification.NEW_MATCH:
                    print("Inserting new tracked object...")
                    currtIndex = self.InsertNewTrackedObject(
                        dBoxesCurrLabel[dIdx], l, dFeaturesCurrLabel[dIdx]
                    )
                    currtUpdated[currtIndex] = True

        # For all others which have not been updated, propagate by
        # movement vector and also decrease life integer by one, down to min
        currtNotUpdated = currtUpdated == False
        self._bboxes[currtNotUpdated] = self.GetPredictedPosition(
            self._bboxes[currtNotUpdated], self._movementVecs[currtNotUpdated]
        )
        self._lives[currtNotUpdated] = self.UpdateLifeIndex(
            self._lives[currtNotUpdated], False
        )

        # Cleanup those which have min life
        self._trackingIDs[self._lives == self._minLife] = 0 

        self.PrintStatus()


    def ExtractFeatures(self, image : np.ndarray, bboxes : np.ndarray) -> np.ndarray:
        # Dummy feature extraction
        # TODO
        shape = bboxes.shape
        #print(shape)
        return np.ones((shape[0],self._featureSpaceSize))
  
    def GetPredictedPosition(self, boxes : np.ndarray, 
                             movementVec : np.ndarray) -> np.ndarray:
        if(len(boxes)==0): return boxes
        boxes_movementVecs = np.hstack((boxes, movementVec))
        traslatedBoxes = np.apply_along_axis(
            lambda x: BoundingBox.Traslate(x[:4], x[4:6]), \
                1, boxes_movementVecs
        )
        return traslatedBoxes

    def ComputeDistancesMatrix(self, centersA : np.ndarray, centersB : np.ndarray):
        #  Compute euclideal distance matrix between 2D points
        distancesMatrix = distance_matrix(centersA, centersB)
        return distancesMatrix
        

    def InsertNewTrackedObject(self, box : np.ndarray, label : int, 
                                features : np.ndarray) -> int:
        # Check if space is present for inserting a tracking object
        maskAvailable = self._lives == 0
        currIndex = GetNthOccurenceIndex(maskAvailable, 0)

        if(currIndex >= 0):
            # Space is available for tracking one more object
            self._bboxes[currIndex] = box
            self._labels[currIndex] = label
            self._features[currIndex] = features
            self._movementVecs[currIndex] = np.zeros(2)
            self._trackingIDs[currIndex] = self.GetNextUniqueTrackID()
            self._lives[currIndex] = self._minLife + 1
        return currIndex

    def GetNextUniqueTrackID(self) -> int:
        # Tracking ID is unique because always incremental
        return np.max(self._trackingIDs) + 1

    # Get min index of a single slice
    def GetDetectedMatchClassification(self, slice : np.ndarray) -> np.ndarray:
        # Get min value
        sortedArray = np.argsort(slice)
        minIndex = sortedArray[0]
        minValue = slice[minIndex]

        # Get second min value
        secondMinValue = 1e9
        if(len(sortedArray)> 1):
            secondMinValue = abs(minValue - slice[sortedArray[1]])
        
        # Estimate match status based on extracted metrics
        if(minValue < 100):
            if(secondMinValue > 50):
                return np.array([MatchClassification.CORRESPONDENT, minIndex])
            else:
                return np.array([MatchClassification.OCCLUSION, minIndex])
        else:
            return np.array([MatchClassification.NEW_MATCH, -1])

    def UpdateLifeIndex(self, lifeIndex : int, isIncrement : bool) -> int:
        if isIncrement:
            lifeIndexRes = lifeIndex + 1
        else:
            lifeIndexRes = lifeIndex - 1
        return np.clip(lifeIndexRes, self._minLife, self._maxLife)

    def PrintStatus(self) -> None:
        print('== TRACKED OBJECTS ==')
        print('Boxes:')
        print(self._bboxes)
        print('Classes:')
        print(self._labels)
        print('Features:')
        print(self._features)
        print('Movement vectors:')
        print(self._movementVecs)
        print('Tracking IDs:')
        print(self._trackingIDs)
        print('Lives:')
        print(self._lives)
        print('===================')

    def GetTrackedObjects(self, minLife = 3) -> dict:
        
        selectionMask = self._lives >= minLife

        bboxes = self._bboxes[selectionMask]
        labels = self._labels[selectionMask]
        ids = self._trackingIDs[selectionMask]

        trackedPredictions = {
            'boxes': bboxes,
            'labels': labels,
            'ids': ids
        }

        return trackedPredictions
