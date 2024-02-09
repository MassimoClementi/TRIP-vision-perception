# Date:     2024-02-08
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Bounding box utility functions

import numpy as np

def GetCenter(box : np.ndarray) -> np.ndarray:
    res = np.array(((box[2]+box[0])/2, (box[3]+box[1])/2))
    return res

def Traslate(box : np.ndarray, shift : np.ndarray) -> np.ndarray:
    res = np.array((
        (box[0] + shift[0]),
        (box[1] + shift[1]),
        (box[2] + shift[0]),
        (box[3] + shift[1])
        ))
    return res

