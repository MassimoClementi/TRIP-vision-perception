import numpy as np
import cv2
from Framegrabber import Framegrabber

# framegrabber_path = 0
framegrabber_path = "../videos/traffic-short.mp4"
# framegrabber_path = "../videos/roomba-POV.mp4"
# framegrabber_path = "../videos/roomba-POV-2.mp4"
framegrabber = Framegrabber(framegrabber_path)
framegrabber.set_scaling_factor(1.0)
framegrabber.set_sampling_interval(1)


cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)


def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      
      # put coordinates as text on the image
      cv2.putText(frame, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
      # draw point on the image
      cv2.circle(frame, (x,y), 3, (0,255,255), -1)

cv2.setMouseCallback('Input', click_event)


radius = 5

# Define scene keypoints
# top_left = (260, 311)
# top_right = (500, 320)
# bottom_right = (410, 715)
# bottom_left = (5, 625)
top_left = (350, 28)
top_right = (425, 28)
bottom_right = (480, 190)
bottom_left = (330, 200)

keypoints = [top_left, top_right, bottom_right, bottom_left]

prj_size = (200, 500) # W x H
projection_keypoints = [
    (0, 0),
    (prj_size[0], 0),
    (prj_size[0], prj_size[1]),
    (0, prj_size[1])
]

matrix = cv2.getPerspectiveTransform(
    np.float32(keypoints), 
    np.float32(projection_keypoints)
)

# Grab first frame
frame = framegrabber.grab_frame()
frameCount = framegrabber.get_frame_count()

while not framegrabber.is_ended():

    projection = cv2.warpPerspective(frame, matrix, prj_size)

    for p in keypoints:
        cv2.circle(frame, p, radius, (0,0,255), -1)

    cv2.imshow('Input', frame)
    cv2.imshow('Output', projection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Get next frame
    frame = framegrabber.grab_frame()
    frameCount = framegrabber.get_frame_count()


