# Date:     2024-01-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Client which performs API calls to server

import cv2
import numpy as np
from Framegrabber import Framegrabber
import APIs
import CoreEngine


# Load image
# framegrabber_path = 0
# framegrabber_path = "../images/test-1.jpg"
framegrabber_path = "../images/test-2.jpg"
framegrabber = Framegrabber(framegrabber_path)
framegrabber.set_scaling_factor(0.50)

apis = APIs.RESTAPIs_v1('http://localhost:5000')
objectDetector = CoreEngine.MyObjectDetector()

# Create visualization window(s)
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

# Loop over each frame of the input video
frame = framegrabber.grab_frame()
while not framegrabber.is_ended():

    # predictions = objectDetector.Detect(np.array([frame]), minScore=0.8)
    predictions = apis.DetectObjects(frame)
    print(predictions)

    objectDetector.GetResultsOverlay(frame, predictions[0])
    cv2.imshow('Output', frame)
    # cv2.waitKey(0) # waits until a key is pressed
    # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Get next frame
    frame = framegrabber.grab_frame()

# Release the capture and close all windows
framegrabber.cap_release()


# Call API
# dec = myAPI.PerformGaussianBlur(image)

# Show results
# cv2.imshow("Result",np.hstack((image, dec)))
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing imag


