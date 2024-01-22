# Date:     2024-01-18
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Core engine that defines the logic for performing all the
#           required tasks, both elaboration and visualization

import numpy as np
import cv2
import torch
import torchvision
from COCOLabels import COCOLabels_2017


# def PerformGaussianBlur(Image : np.array) -> np.array:
#     imageProcessed = cv2.GaussianBlur(Image,(5,5),cv2.BORDER_DEFAULT)
#     return imageProcessed


class MyObjectDetector():
    def __init__(self) -> None:
        '''Instantiate an object detector'''
        self.model = None
        self.isModelCreated = False
        self.device = self.GetCUDADeviceOrCPU()

    def GetCUDADeviceOrCPU(self) -> torch.device:
        '''Get device to use with pytorch'''
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Torch is using device:', device)
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory cllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Memory cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        return device
        
    def CreateDNNModel(self):
        '''Create the deep learning model architecture'''
        print('Creating DNN model...')
        # model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone = True)
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False, pretrained_backbone=False
        ).to(self.device)
        self.LoadModelStateDict('./config/fasterrcnn_mobilenet_v3_large_fpn-state-dict.pth')
        self.isModelCreated = True
        return
    
    def SaveModelStateDict(self, path : str):
        '''Save weights and status of neural network to disk'''
        print('Saving DNN state dict to ', path)
        torch.save(self.model.state_dict(), path)
        return
    
    def LoadModelStateDict(self, path : str):
        '''Load weights and status of neural network from disk'''
        print('Loading DNN state dict to ', path)
        self.model.load_state_dict(torch.load(path))
        return

    def Detect(self, images : np.array, minScore : float) -> list:
        '''Detect objects in the image'''
        if(not self.isModelCreated): self.CreateDNNModel()
        
        print('Performing inference on provided samples...')
        self.model.eval()
        x = torch.tensor(
            np.transpose(np.array(images/255, dtype=float), (0, 3, 1, 2))).to(self.device)
        predictions = self.model.double()(x.double())

        print('Selecting matches by score...')
        for p in predictions:
            scores = p['scores']
            mask = scores > minScore
            p['boxes'] = p['boxes'][mask].detach().numpy()
            p['labels'] = p['labels'][mask].detach().numpy()
            p['scores'] = p['scores'][mask].detach().numpy()
        print(predictions)

        return predictions
    
    def GetResultsOverlay(self, image : np.array, predictions : dict) -> np.array:
        '''Display object detection results as overlay'''
        for i in range(len(predictions['boxes'])):
            score = predictions['scores'][i]
            label = predictions['labels'][i]
            c1, r1, c2, r2 = map(int, predictions['boxes'][i].tolist())
            display_color = (0, 255, 0)
            image = cv2.rectangle(image, (c1, r1), (c2, r2), display_color, 3)
            cv2.putText(image, COCOLabels_2017().GetLabel(label), 
                        (c1 + 5, r1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                        display_color, 2)
        return
  