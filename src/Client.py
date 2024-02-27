# Date:     2024-01-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Client which performs API calls to server

import cv2
import numpy as np
from Framegrabber import Framegrabber
import APIs
import CoreEngine
from MultiObjectTracker import MultiObjectTracker
from FeatureExtractors import FeatureExtractorORB
from FeatureMatchers import FeatureMatcherORB

# Load image
# framegrabber_path = 0
# framegrabber_path = "../images/test-1.jpg"
framegrabber_path = "../images/test-2.jpg"
framegrabber = Framegrabber(framegrabber_path)
framegrabber.set_scaling_factor(0.50)
framegrabber.set_sampling_interval(10)

apis = APIs.RESTAPIs_v1('http://localhost:5000')
objectDetector = CoreEngine.MyObjectDetector()

multiObjectTracker = MultiObjectTracker(
    maxNumTrackedObjects=20,
    correspondenceMaxDistance=35,
    occlusionMinDistance=20
)

# Create visualization window(s)
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

# Grab first frame
frame = framegrabber.grab_frame()

# Create video writer for output
fourcc = cv2.VideoWriter_fourcc(*'Mp4v')
video = cv2.VideoWriter('output_video.mp4', fourcc, 25, (frame.shape[1], frame.shape[0]))

# Loop over each frame of the input video
while not framegrabber.is_ended():

    #predictions = objectDetector.Detect(np.array([frame]), minScore=0.8)
    predictions = apis.DetectObjects(frame)
    #print(predictions)
    #print(predictions[0]['boxes'].shape)
    #print(predictions[0]['labels'].shape)

    multiObjectTracker.Update(
        frame,
        predictions[0]['boxes'],
        predictions[0]['labels']
    )
    trackedPredictions = multiObjectTracker.GetTrackedObjects(minLife=3)

    #objectDetector.GetResultsOverlay(frame, frameCount, predictions[0])
    objectDetector.GetResultsOverlay(frame, frameCount, trackedPredictions, useTrackingIDs=True)
    cv2.imshow('Output', frame)
    video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Get next frame
    frame = framegrabber.grab_frame()
    frameCount = framegrabber.get_frame_count()

# Release the capture and close all windows
framegrabber.cap_release()
video.release()

