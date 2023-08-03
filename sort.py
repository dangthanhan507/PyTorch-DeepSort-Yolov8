import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from detector import BBox

def IOU(bbox1, bbox2):
    '''
        Calculate intersection over union.
        Area of overlapping boudning boxes
    '''
    x1TL,y1TL = bbox1.getTL()
    x1BR,y1BR = bbox1.getBR()
    
    x2TL,y2TL = bbox2.getTL()
    x2BR,y2BR = bbox2.getBR()
    
    biggestLX  = max(x1TL, x2TL)
    smallestRX = min(x1BR, x2BR)
    
    biggestYT  = max(y1TL, y2TL)
    smallestYB = max(y1BR, y2BR)
    
    area = (smallestRX - biggestLX) * (smallestYB - biggestYT)
    return area
def bbox_to_state(bbox):
    xTL, yTL = bbox.getTL()
    xBR, yBR = bbox.getBR()
    
    cx = (xTL + xBR) / 2
    cy = (yTL + yBR) / 2
    
    r = (xBR - xTL) / (yBR - yTL)
    s = (yBR - yTL)
    
    return np.array([[cx,cy,s,r,0,0,0]]).T
def state_to_bbox(state):
    cx = state[0,0]
    cy = state[1,0]
    s  = state[2,0]
    r  = state[3,0]
    
    w  = r*s
    h  = w/r
    
    xTL = cx - w/2
    xBR = cx + w/2

    yTL = cy - h/2
    yBR = cy + h/2

    return BBox(xTL,yTL,xBR,yBR,'unknown')

def calc_velocity(curr_state, prev_state):
    vel = curr_state[:3] - prev_state[:3]
    return vel

class KalmanFilter:
    '''
    x_k+1 = A*x_k + N(0,Q)
    y_k   = C*x_k + N(0,R)
    '''
    def __init__(self, A, C, Q, R,obj_id=0):
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        
        self.id = obj_id
        
        self.state = None
        self.cov   = None
        
    def initialize(self, state0, cov0):
        self.state = state0
        self.cov   = cov0
    def predict(self):
        self.state  = self.A @ self.state
        self.cov = self.A @ self.cov @ self.A.T + self.Q
    def update(self, meas):
        K = self.cov @ self.C.T @ np.linalg.inv(self.C @ self.cov @ self.C.T + self.R)
        self.state  = self.state + K @ (meas - self.C @ self.state)
        self.cov = (np.eye(self.cov.shape[0]) - K @ self.C) @ self.cov
        
    def drawState(self, image):
        bbox = state_to_bbox(self.state)
        bbox.name = f'ID:{self.id}'
        
        height,width,_ = image.shape
        thick = int((height+width)//900)
        cv2.rectangle(image, (int(bbox.x0),int(bbox.y0)), (int(bbox.x1),int(bbox.y1)), (255,0,0))
        cv2.putText(image, bbox.name, (int(bbox.x0),int(bbox.y0)-12), 0, 1e-3*height, (255,0,0), thick//3)
        return image
    
class SORT:
    '''
    representing objects as [u,v,s,r,udot,vdot,sdot,rdot]
    u,v = x,y image coords
    s = scale
    r = aspect ratio (ratio) w/h
    '''
    def __init__(self):
        '''
            SORT steps:
            ===========
            -> perform detection
            -> run prediction step on all object KF
            -> associate new detections w/ bboxes from predicted object KF
            -> perform KF update step on all associated detections
            -> any unassociated objects get deleted
            
            Things to note:
            ===============
            -> maintain an id counter
            -> increment id counter for every detection and associate object w/ id
            -> velocity of new object is 0.
            -> once objects are associated, you treat the new detections as the "measurement"
                -> subtract center points of both bboxes to get velocity
                -> keep everything else as is in order to perform update step.
        '''
        self.id_ctr = 0
        self.objs = []
        
    def initialize_object(self, bbox):
        track_vector = bbox_to_state(bbox)
        track_cov    = np.eye(7)
        
        #assign id
        id_ = self.id_ctr
        self.id_ctr += 1
        
        A = np.eye(7)
        #first 3 rows, last 3 column
        A[:3,4:] = np.eye(3)
        
        obj = KalmanFilter(A=A, C=np.eye(7), Q=np.eye(7), R=1e-3*np.eye(7),obj_id=id_)
        obj.initialize(track_vector, track_cov)
        self.objs.append(obj)
        
    def track_boxes(self, bboxes):
        #predict obj phase
        for obj in self.objs:
            obj.predict()        
        
        #associate
        obj_matched, det_matched = self.hungarian(bboxes)
        print('Found these matches: ', len(obj_matched))
        #any unmatched, just leave out....
        #perform update step 
        objs_new = []
        for i in range(len(obj_matched)):
            obj_idx = obj_matched[i]
            det_idx = det_matched[i]
            
            det = bbox_to_state(bboxes[det_idx])
            det[4:] = calc_velocity(det, self.objs[obj_idx].state)
            
            self.objs[obj_idx].update(det)
            objs_new.append(self.objs[i])
        
        #filter only objects that got tracked
        self.objs = objs_new
        
        #initialize any unmatched detections as a new track
        for i in range(len(bboxes)):
            if i not in det_matched:
                self.initialize_object(bboxes[i])
    def drawTrack(self, image):
        for obj in self.objs:
            image = obj.drawState(image)
        return image
        
    def hungarian(self, bboxes):
        '''
        In order to run hungarian, we create a cost matrix by hand.
        Then we run it through a solver to find the best solution.
        Metric for Hungarian is IOU
        '''
        predicted_boxes = []
        for obj in self.objs:
            bbox = state_to_bbox(obj.state)
            predicted_boxes.append(bbox)
        
        #compare predicted boxes (rows) vs measured boxes (cols)
        
        m = len(predicted_boxes)
        n = len(bboxes)
        cost_matrix = np.zeros((m,n))
        for pred_idx in range(m):
            for meas_idx in range(n):
                cost_matrix[pred_idx,meas_idx] = IOU(predicted_boxes[pred_idx], bboxes[meas_idx])
        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        obj_matched = []
        det_matched = []
        for i in range(len(row_ind)):
            obj_idx = row_ind[i]
            det_idx = col_ind[i]
            if cost_matrix[obj_idx,det_idx] >= 100:
                obj_matched.append(obj_idx)
                det_matched.append(det_idx)
        
        return obj_matched, det_matched