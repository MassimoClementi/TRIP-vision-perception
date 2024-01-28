# Date:     2024-01-19
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Definition of the REST APIs which can be used to
#           communicate with the processing server

import numpy as np
from EncoderDecoder import EncoderDecoderNumpy, EncoderDecoderImage
import requests
from Utilities import print_execution_time

class RESTAPIs_v1():

    def __init__(self, url) -> None:
        '''Initialize REST API interface'''
        self.url = url
        self.version = 'v1.0'
        self.GetServerInformation()
        return

    def GetServerInformation(self) -> bool:
        '''Send status request to server and check availability'''
        print('Getting server information...')
        resultJson = requests.get(url=self.url).json()
        print(resultJson)
        assert resultJson['running'] == True
        assert self.version in resultJson['supportedAPIs']
        return

    @print_execution_time
    def DetectObjects(self, image : np.array):
        '''Detect objects on the image using FasterRCNN model'''
        # Create API request
        enc = EncoderDecoderImage().Encode(image, np.uint8)
        requestJson = {
            'image': enc
        }

        # Call API with request and get results
        print('Performing REST API call...')
        resultJson = requests.post(
                    self.url + "/api/v1.0/detectobjects",
                    json = requestJson
                ).json()
        print('Response acquired')

        # Format results
        prediction = []
        for p in resultJson:
            prediction.append({
                'boxes': EncoderDecoderNumpy().Decode(p['boxes'], np.float32),
                'labels': EncoderDecoderNumpy().Decode(p['labels'], np.float32),
                'scores': EncoderDecoderNumpy().Decode(p['scores'], np.float32)
            })
        return prediction
