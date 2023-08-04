import cv2
import torch
from ultralytics import YOLO


class BBox:
    def __init__(self, x0, y0, x1, y1, name):
        '''
        FORMAT:
        -------
            -> we follow top left bottom right format. (xyxy)
            -> (x0,y0) reprsents top left point of box
            -> (x1,y1) represents bottom right point of box
            -> with this in mind in the image context, we know that x0 < x1 and y0 < y1.
            -> we also have a name attached to distinguish between bounding boxes.

        Parameters:
        -----------
            x0: double
            x1: double
            y0: double
            y1: double
            name: string representing class name for model
        '''
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.name = name
        # assert x1 - x0 != 0 and y1 - y0 != 0

    def getTL(self):
        # get topleft xy point of bounding box
        return (self.x0,self.y0)
    def getBR(self):
        # get bottom right xy point of bounding box
        return (self.x1,self.y1)
    
    def drawBox(self, image):
        '''
        Description:
        -------------
        visualizes a bounding box by drawing it on an image.
        this happens in place as to not use up more memory.

        Parameters:
        ------------
            image: (M,N,3) np.uint8 array representing RGB image.

        Returns:
        --------
            drawnImage: (M,N,3) np.uint8 array representing RGB image with a bounding box drawn on it.
        '''
        height,width,_ = image.shape
        thick = int((height+width)//900)
        cv2.rectangle(image, (int(self.x0),int(self.y0)), (int(self.x1),int(self.y1)), (255,0,0))
        cv2.putText(image, self.name, (int(self.x0),int(self.y0)-12), 0, 1e-3*height, (255,0,0), thick//3)
        return image

    def getImagePatch(self, image):
        patch = image[int(self.y0): int(self.y1), int(self.x0): int(self.x1)]
        # print(patch.shape, self.x0, self.x1)
        assert patch.shape[0] != 0 and patch.shape[1] != 0
        return patch
    
class YoloDetector:
    def __init__(self, model_file):
        if not model_file.endswith('.pt'):
            raise ValueError("File string should end with .pt")
        
        self.model = YOLO(model_file)
    def eval(self, image, filter_cls = []):
        '''
        Description:
        -------------
        Evaluates a colored image using yolo and takes out detection.

        Parameters:
        ------------
            image: (M,N,3) np.uint8 array representing RGB image.
            filter_cls: [String] representing list of strings that we would filter our results by 
                        to only focus on specific classes on a model.
        Returns:
        --------

            box_ret: [BBox] returns list of BBox containing results
        '''
        results = self.model.predict(image,conf=0.5)
        box_ret = []
        '''
        Yolov8 NOTE:
        -------------
        Model inference on yolov8 returns a results.

        Results contains boxes which is what we need.
        Boxes contains xyxy (top left to bottom right)

        '''
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist() # top left, bot right
                c = box.cls #class index

                #use class index to access name in models
                name = self.model.names[int(c)]

                #if we decide to filter boxes for a specific kind of class
                if len(filter_cls) > 0:
                    if name in filter_cls:
                        box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
                else:
                    box_ret.append(BBox(xyxy[0],xyxy[1],xyxy[2],xyxy[3],name))
        return box_ret
