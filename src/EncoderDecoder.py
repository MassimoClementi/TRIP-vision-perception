# Date:     2024-01-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Encode an decode methods to transform images and multidimensional
#           arrays to text and the other way around. It will allow, in our case,
#           to embed the data in the HTTP POST request of a REST API

import numpy as np
import base64
import cv2
import json
from Utilities import print_execution_time


# Convert OpenCV images to a JSON serialized representation and 
# the other way around.
class EncoderDecoderImage():

    def __init__(self, imgtype = '.tiff') -> None:
        self.imgtype = imgtype

    @print_execution_time
    def Encode(self, data : np.array, dtype : type) -> str:
        print('Encoding data...')
        _, buffer = cv2.imencode(self.imgtype, np.array(data, dtype=dtype))
        encoded_as_text = base64.b64encode(buffer)
        return encoded_as_text.decode('utf-8')
    
    @print_execution_time
    def Decode(self, dataEncoded : str, dtype : type) -> np.array:
        print('Decoding data...')
        bytesArray = dataEncoded.encode('utf-8')
        encoded_text = base64.b64decode(bytesArray)
        encoded_as_np = np.frombuffer(encoded_text, dtype=dtype)
        data_buffer = cv2.imdecode(encoded_as_np, flags=1)
        return data_buffer

# Numpy array JSON serialization
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Convert numpy ndarrays to a JSON serialized representation and 
# the other way around.
class EncoderDecoderNumpy():

    @print_execution_time
    def Encode(self, data : np.array, dtype : type) -> str:
        print('Encoding data...')
        json_dump = json.dumps({'data': data}, 
            cls=NumpyEncoder)
        return json_dump
    
    @print_execution_time
    def Decode(self, dataEncoded : str, dtype : type) -> np.array:
        print('Decoding data...')
        json_load = json.loads(dataEncoded)
        a_restored = np.asarray(json_load["data"])
        return a_restored    

