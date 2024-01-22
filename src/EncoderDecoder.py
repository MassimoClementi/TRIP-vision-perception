# Date:     2024-01-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Multidimensional arrays encoding and decoding methods

import numpy as np
# import base64
# import cv2
import json


# Numpy array JSON serialization
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Convert data to a JSON serialized representation and 
# the other way around. This allows to embed the data in the
# HTTP POST request, for instance of a REST API
class EncoderDecoder_v1():

    def Encode(self, data : np.array, dtype : type) -> str:
        print('Encoding data...')
        # _, buffer = cv2.imencode('.tiff', np.array(data, dtype=dtype))
        # encoded_as_text = base64.b64encode(buffer)
        # return encoded_as_text.decode('utf-8')
        # print(data.shape)
        json_dump = json.dumps({'data': data}, 
            cls=NumpyEncoder)
        return json_dump
    
    def Decode(self, dataEncoded : str, dtype : type) -> np.array:
        print('Decoding data...')
        # bytesArray = dataEncoded.encode('utf-8')
        # encoded_text = base64.b64decode(bytesArray)
        # encoded_as_np = np.frombuffer(encoded_text, dtype=dtype)
        # data_buffer = cv2.imdecode(encoded_as_np, flags=1)
        # return data_buffer
        json_load = json.loads(dataEncoded)
        a_restored = np.asarray(json_load["data"])
        # print(a_restored)
        # print(a_restored.shape)
        return a_restored    

