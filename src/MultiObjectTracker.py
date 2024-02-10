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

        # Debug: insert arbitrary tracked objects 
        # res = self.InsertNewTrackedObject(
        #     np.array([[10,10,20,20]]), 64, np.ones(self._featureSpaceSize)
        # )
        # print(res)
        # res = self.InsertNewTrackedObject(
        #     np.array([[100,200,300,400]]), 64, np.ones(self._featureSpaceSize)
        # )
        # print(res)
        # ------

        self.PrintStatus()

        currtUpdated =  np.zeros(self._maxNumTrackedObjects, dtype=bool)

        # For each unique class of detected matches
        for l in np.unique(dLabels):
            print('Current label is', l)

            # Between detected objects, select those which are of given class
            dCurrLabelMask = dLabels == l
            dCurrLabelIndexes = GetIndexesFromMask(dCurrLabelMask)
            dBoxesCurrLabel = dBoxes[dCurrLabelMask]

            # Between tracked object, select those which are of given class
            tCurrLabelMask = self._labels == l
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
            #print(currPredictedTrackedBoxes)

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
                #print(dCentersCurrLabel)
                tPredictedCentersCurrLabel = np.apply_along_axis(
                    BoundingBox.GetCenter, 1, tPredictedBoxesCurrLabel
                )
                #print(tPredictedCentersCurrLabel)
                matrixCentersDistances = self.ComputeDistancesMatrix(
                    dCentersCurrLabel, tPredictedCentersCurrLabel
                )
                matrixFeaturesDistances = self.ComputeDistancesMatrix(
                    dFeaturesCurrLabel, tFeaturesCurrLabel
                )
            #distancesMatrixTEST = self.ComputeDistancesMatrix(currBoxes, currBoxes)
            #print(matrixCentersDistances)
            #print(matrixFeaturesDistances)

            # Based of center distances and features distances, classify each
            # match as either:
            # - correspondent (get corresponding tracked object index)
            # - occlusion (get corresponding tracked object index)
            # - new match
            weightFactor = 1.0
            matrixWeightedDistances = weightFactor * matrixCentersDistances + \
                (1.0 - weightFactor) * matrixFeaturesDistances
            #print(avgDistancesMatrix)
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
            # TODO

            #dIndexArr = np.array([dCurrLabelIndexes]).T
            #dMatchesClassificationAppended = np.hstack(
            #    (dIndexArr, dMatchesClassification)
            #)
            #print(dMatchesClassificationAppended)
            #print(dMatchesClassificationAppended.shape)

            for dIdx in range(dMatchesClassification.shape[0]):
                currClassRes = dMatchesClassification[dIdx][0]
                currCorrespondIndex = dMatchesClassification[dIdx][1]
                
                if currClassRes == MatchClassification.CORRESPONDENT:
                    currtIndex = tCurrLabelIndexes[currCorrespondIndex]
                    print("Updating tracked object at index", currtIndex, "...")
                    currtUpdated[currtIndex] = True
                    self._bboxes[currtIndex] = dBoxesCurrLabel[dIdx]
                    self._lives[currtIndex] =+ self.UpdateLifeIndex(
                        self._lives[currtIndex], True
                    )
                if currClassRes == MatchClassification.OCCLUSION:
                    print("Tracked object at index", currtIndex, "in occlusion")
                    currtIndex = tCurrLabelIndexes[currCorrespondIndex]
                    currtUpdated[currtIndex] = True
                    pass
                if currClassRes == MatchClassification.NEW_MATCH:
                    print("Inserting new tracked object...")
                    currtIndex = self.InsertNewTrackedObject(
                        dBoxesCurrLabel[dIdx], l, dFeaturesCurrLabel[dIdx]
                    )
                    currtUpdated[currtIndex] = True

        # Finally, for all others, propagate by movement vector and also decrease 
        # life integer by one, down to life min
        currtNotUpdated = currtUpdated == False
        self._lives[currtNotUpdated] = self.UpdateLifeIndex(
            self._lives[currtNotUpdated], False
        )

        self.PrintStatus()


    def ExtractFeatures(self, image : np.ndarray, bboxes : np.ndarray) -> np.ndarray:
        # Dummy feature extraction
        # TODO
        shape = bboxes.shape
        #print(shape)
        return np.ones((shape[0],self._featureSpaceSize))
  
    def GetPredictedPosition(self, boxes : np.ndarray, 
                             movementVec : np.ndarray) -> np.ndarray:
        # Dummy predicted position
        # TODO
        #print(boxes)
        #print(boxes.shape)
        #print(movementVec)
        #print(movementVec.shape)
        return boxes

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

    def UpdatePositionPrediction(self):
        pass



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

        print(trackedPredictions)

        return trackedPredictions
