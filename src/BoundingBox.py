# Date:     2024-02-08
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Bounding box utility functions

import numpy as np

def GetCenter(box : np.ndarray) -> np.ndarray:
    res = np.array(((box[2]+box[0])/2, (box[3]+box[1])/2))
    return res

def Traslate(box : np.ndarray, shift : np.ndarray) -> np.ndarray:
    res = np.array([
        (box[0] + shift[0]),
        (box[1] + shift[1]),
        (box[2] + shift[0]),
        (box[3] + shift[1])
        ])
    return res

def GetDelta(bboxB : np.ndarray, bboxA : np.ndarray):
    centerB = GetCenter(bboxB)
    centerA = GetCenter(bboxA)
    centerDelta = centerB - centerA
    return centerDelta

def GetImagePatch(box : np.ndarray, image : np.ndarray) -> np.ndarray:
    c1, r1, c2, r2 = np.asarray(box, dtype=int)
    patch = image[r1:r2,c1:c2,:]
    return patch