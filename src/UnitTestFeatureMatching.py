# Date:     2026-04-07
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Test feature extraction and matching

import cv2
import numpy as np
from Framegrabber import Framegrabber
from FeatureExtractors import FeatureExtractorORB
from FeatureMatchers import FeatureMatcherORB

# Define images to use
frameA = cv2.imread('../images/test-tower-1.jpeg')
frameB = cv2.imread('../images/test-tower-2.jpeg')

# Define feature extractors
featureExtractorA = FeatureExtractorORB()
featureExtractorB = FeatureExtractorORB()

# Extract features from frame A
featureExtractorA.ComputeFeatures(frameA)
frameDisplay = featureExtractorA.GetKeypointsVisualization()
cv2.imshow('Output', frameDisplay)
cv2.waitKey(0)

# Extract features from frame B
featureExtractorB.ComputeFeatures(frameB)
frameDisplay = featureExtractorB.GetKeypointsVisualization()
cv2.imshow('Output', frameDisplay)
cv2.waitKey(0)

# Define feature matcher
featureMatcherORB = FeatureMatcherORB()

# Perform feature matching
featureMatcherORB.ComputeMatchingFeatures(featureExtractorA, featureExtractorB)
print('Matching loss value:', featureMatcherORB.GetMatchingLoss())
frameDisplay = featureMatcherORB.GetMatchesVisualization()
cv2.imshow('Output', frameDisplay)
cv2.waitKey(0)