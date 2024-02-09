# Date:     2024-02-08
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define a multiobjects tracker

import numpy as np
from COCOLabels import COCOLabels_2017
from Utilities import print_execution_time
from scipy.spatial import distance_matrix


class MultiObjectTracker():

    def __init__(self, maxNumTrackedObjects : int) -> None:
        self._maxNumTrackedObjects = maxNumTrackedObjects
        self._minLife = 5
        self._maxLife = 5

        self._featureSpaceSize = 10

        # Initialize data arrays
        self._bboxes = np.zeros((self._maxNumTrackedObjects, 4))
        self._classes = np.zeros(self._maxNumTrackedObjects)
        self._features = np.zeros((self._maxNumTrackedObjects, self._featureSpaceSize))
        self._movementVecs = np.zeros((self._maxNumTrackedObjects, 2))
        self._trackingIDs = np.zeros(self._maxNumTrackedObjects)
        self._lifes =  np.zeros(self._maxNumTrackedObjects)

    @print_execution_time
    def Update(self, image : np.ndarray, bboxes : np.ndarray, classes : np.ndarray) -> None:

        res = self.InsertNewTrackedObjects(
            np.array([[10,10,20,20]]), 64, np.ones(self._featureSpaceSize)
        )
        print(res)
        res = self.InsertNewTrackedObjects(
            np.array([[100,200,300,400]]), 64, np.ones(self._featureSpaceSize)
        )
        print(res)

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
            currPredictedTrackedBoxes = self.GetPredictedPosition(currTrackedBoxes, currMovementVecs)

            # Extract matrix of distances between each detected match of given 
            # class and each prediction of tracked match of given class
            # Then compute distance between features
            if(currPredictedTrackedBoxes.shape[0] == 0):
                nsize = currBoxes.shape[0]
                centersDistancesMatrix = np.ones((nsize,nsize)) * 1e9
                featuresDistancesMatrix = np.ones((nsize,nsize)) * 1e9
            else:
                detCenters = np.apply_along_axis(self.ComputeBoxesCenter, 1, currBoxes)
                trackedCenters = np.apply_along_axis(self.ComputeBoxesCenter, 1, currPredictedTrackedBoxes)
                centersDistancesMatrix = self.ComputeDistancesMatrix(currBoxes, currPredictedTrackedBoxes)
                featuresDistancesMatrix = self.ComputeDistancesMatrix(currFeatures, currTrackedFeatures)
            #distancesMatrixTEST = self.ComputeDistancesMatrix(currBoxes, currBoxes)
            print(centersDistancesMatrix)
            print(featuresDistancesMatrix)


            # Based of features and distances, determine which matches are 
            # correspondent, which are new matches and which are in occlusion
            # TODO



        # Finally, with respect to all tracked objects
		# - for those which are correspondent, compute movement vector, 
        #   update box, features and add life integer by one, up to max
		# - for those which are new matches, update box, features, movement 
        #   vector as zeros and add life integer by one, up to max
		# - for those which are in occlusion, propagate box by movement vector 
        #   but do not change life integer 
        # - for all others, propagate by movement vector and also  decrease 
        #   life integer by one, down to life min
        # TODO
            
        



    def ExtractFeatures(self, image : np.ndarray, bboxes : np.ndarray) -> np.ndarray:
        # Dummy feature extraction
        shape = bboxes.shape
        print(shape)
        return np.ones((shape[0],self._featureSpaceSize))
  
    def GetPredictedPosition(self, boxes : np.ndarray, movementVec : np.ndarray) -> np.ndarray:
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
        first_occurence = np.where(maskAvailable)

        if(len(first_occurence[0]) > 0):
            # Space is available for tracking one more object
            currIndex = first_occurence[0][0]
            self._bboxes[currIndex] = box
            self._classes[currIndex] = label
            self._features[currIndex] = features
            self._movementVecs[currIndex] = np.zeros(2)
            self._trackingIDs[currIndex] = 1
            self._lifes[currIndex] = 1
            return True
        else:
            # No space is left for tracking object
            return False

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
  

    def ComputeBoxesCenter(self, box : np.ndarray) -> np.ndarray:
        res = np.array(((box[2]+box[0])/2, (box[3]+box[1])/2))
        return res
    
    def TraslateBox(self, box : np.ndarray, shift : np.ndarray) -> np.ndarray:
        res = np.array((
            (box[0] + shift[0]),
            (box[1] + shift[1]),
            (box[2] + shift[0]),
            (box[3] + shift[1])
        ))
        return res
