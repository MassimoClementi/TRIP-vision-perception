# Date:     2023-01-14
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Website that manages API calls

from flask import Flask, jsonify, request
import numpy as np
from EncoderDecoder import EncoderDecoder_v1
from CoreEngine import MyObjectDetector

app = Flask(__name__)
objectDetector = MyObjectDetector()


# Return server status and features
@app.route('/')
def EndpointServerStatus():
    response = {
        'running': True,
        'supportedAPIs': GetSupportedAPIVersions(),
        'description': 'TRIP Vision Perception elaboration server'
    }
    return response

def GetSupportedAPIVersions():
    versions = ['v1.0']
    return versions

# Request JSON -> response JSON
# @app.route('/api/v1.0/process', methods=['POST'])
# def EndpointGaussianBlur():
#     req = request.get_json()

#     imageEncoded = req['image']
#     image = EncoderDecoder_v1().Decode(imageEncoded, np.uint8)
#     print(image.shape)

#     imageProcessed = PerformGaussianBlur(image)

#     imageRensponseEncoded = EncoderDecoder_v1().Encode(imageProcessed, np.uint8)
#     response = {
#         'image': imageRensponseEncoded
#     }

#     return response


@app.route('/api/v1.0/detectobjects', methods=['POST'])
def EndpointDetectObjects():
    req = request.get_json()

    imageEncoded = req['image']
    image = EncoderDecoder_v1().Decode(imageEncoded, np.uint8)

    predictions = objectDetector.Detect(np.array([image]), minScore=0.8)

    response = []
    for p in predictions:
        response.append ({
            'boxes': EncoderDecoder_v1().Encode(p['boxes'], np.float32),
            'labels': EncoderDecoder_v1().Encode(p['labels'], np.float32),
            'scores': EncoderDecoder_v1().Encode(p['scores'], np.float32)
        })

    return response


if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host='0.0.0.0')
