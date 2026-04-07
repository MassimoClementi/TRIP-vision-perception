import numpy as np
import cv2
from Framegrabber import Framegrabber
from MarkerDetectors import ArucoMarkersDetector, ArucoDetectionResults

# Define inputs
framegrabber = Framegrabber('../videos/aruco-marker.mov')
# framegrabber = Framegrabber(0)
framegrabber.set_scaling_factor(1.0)
framegrabber.set_sampling_interval(1)

# Get windows
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Projection', cv2.WINDOW_NORMAL)

# Define Aruco markers to detect 
arucoMarkersDetector = ArucoMarkersDetector(cv2.aruco.DICT_6X6_1000)

# Define size of projection area
projSizeLen = 1000
projSize = np.array(np.ones(2) * projSizeLen).astype(int)
print(projSize)

# Specify marker length, scale by projection area
markerLength = 100
objPoints = np.array([
    [- 1,  1],
    [  1,  1],
    [  1, -1],
    [- 1, -1]
]) * 0.5 * markerLength
objPoints = objPoints + projSizeLen/2.0

# Grab first frame
frame = framegrabber.grab_frame()
frameCount = framegrabber.get_frame_count()

prevMatrices = []

while not framegrabber.is_ended():

    # Search markers
    detectionResults = arucoMarkersDetector.DetectMarkers(frame)
    overlayImage = detectionResults.GetResultsOverlay(frame)
    cv2.imshow('Image', overlayImage)

    # If marker has been found
    if(len(detectionResults.corners) > 0):

        # Compue and append transformation matrix
        matrix = cv2.getPerspectiveTransform(
            np.float32(detectionResults.corners[0]), 
            np.float32(objPoints)
        )
        prevMatrices.append(matrix)

        # Compute weighted average of matrix 
        matrixAvg = matrix
        sizeBuffer = 5
        if(len(prevMatrices) >= sizeBuffer):
            prevMatricesNumpy = np.array(prevMatrices[-sizeBuffer:])
            windowFunction = np.blackman(sizeBuffer*2 - 1)[:sizeBuffer]
            weights = windowFunction / np.sum(windowFunction)
            matrixAvg = np.average(prevMatricesNumpy, axis=0, weights=weights)

        # Apply perspective transformation to current frame 
        projection = cv2.warpPerspective(frame, matrixAvg, projSize)
        cv2.imshow('Projection', projection)
    
    waitTimeMs = 1
    if cv2.waitKey(waitTimeMs) & 0xFF == ord('q'):
        break

    # Grab next frame
    frame = framegrabber.grab_frame()
    frameCount = framegrabber.get_frame_count()

    # while cv2.waitKey(1) & 0xFF != ord('q'):
    #    pass


