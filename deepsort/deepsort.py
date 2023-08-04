from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from torchvision import transforms as T

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
    def __init__(self, A, C, Q, R, obj_id):
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

class TrackObj(KalmanFilter):
    def __init__(self, A, C, Q, R, obj_id=0, age=0, matched_before=-1):
        super().__init__(A, C, Q, R, obj_id) 
        self.last_age_matched = matched_before
        self.age = age 
    
class DeepSORT:
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
        self.conf = 0.95 # confidence interval??

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siamese_net = torch.load("ckpts/model640.pt", map_location=device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((128, 128))
        ])
        self.gate_matrix_thresh1 = 9.4877
        self.gate_matrix_thresh2 = 0.85
        self.MAX_AGE = 20
        
    def initialize_object(self, bbox):
        track_vector = bbox_to_state(bbox)
        track_cov    = np.eye(7)
        
        #assign id
        id_ = self.id_ctr
        print("ASSIGNING ID:", id_)
        self.id_ctr += 1
        
        A = np.eye(7)
        #first 3 rows, last 3 column
        A[:3,4:] = np.eye(3)
        
        obj = TrackObj(A=A, C=np.eye(7), Q=np.eye(7), R=1e-3*np.eye(7),obj_id=id_)
        obj.initialize(track_vector, track_cov)
        self.objs.append(obj)
        
    def track_boxes(self, bboxes, image):
        #predict obj phase
        for obj in self.objs:
            obj.predict()        
        
        #associate
        obj_matched, det_matched, unmatched = self.matching_cascade(bboxes, image)
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
            objs_new.append(self.objs[obj_idx])


        # remove any unmatched
        for obj in self.objs:
            if obj not in objs_new \
                    and not (obj.age == 3 and obj.last_age_matched == -1) \
                    and not (obj.age - obj.last_age_matched >= self.MAX_AGE):
                objs_new.append(obj)
        
        #filter only objects that got tracked
        self.objs = objs_new
        
        #initialize any unmatched detections as a new track
        for i in range(len(bboxes)):
            if i not in det_matched:
                self.initialize_object(bboxes[i])

        # update age
        for i in self.objs:
            i.age += 1

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

    def matching_cascade(self, bboxes, image):
        predicted_boxes = []
        for obj in self.objs:
            bbox = state_to_bbox(obj.state)
            predicted_boxes.append(bbox)

        # create cost matrix
        m = len(predicted_boxes)
        n = len(bboxes)
        cost_matrix = np.zeros((m, n))
        d1_matrix = np.zeros((m, n))
        d2_matrix = np.zeros((m, n))
        for pred_idx in range(m):
            for meas_idx in range(n):
                d1 = self.mahalanobis(
                        predicted_boxes[pred_idx], 
                        bboxes[meas_idx], 
                        self.objs[pred_idx])
                d1_matrix[pred_idx, meas_idx] = d1
                # print("d2 predicted boxes:", predicted_boxes[pred_idx].x0, predicted_boxes[pred_idx].x1)
                d2 = self.cosine_distance(
                        predicted_boxes[pred_idx].getImagePatch(image), 
                        bboxes[meas_idx].getImagePatch(image))
                # if d1 < self.gate_matrix_thresh1:
                #     print("Mahalanobis dist is good!!!")
                # if d2 < self.gate_matrix_thresh2:
                #     print("Cosine dist is good!!!")
                # print(f"{pred_idx} {meas_idx}")
                # print("Mahalanobis:", d1)
                # print("cosine:", d2)
                # print("--------------")
                # if d1 <= self.gate_matrix_thresh1 and d2 <= self.gate_matrix_thresh2:
                #     print("this gate matrix should be a 1 here")
                d2_matrix[pred_idx, meas_idx] = d2
                dist_measurement = self.conf * d1 + (1 - self.conf) * d2
                cost_matrix[pred_idx, meas_idx] = dist_measurement

        # create gate matrix
        gate_matrix = np.zeros((m, n))
        for pred_idx in range(m):
            for meas_idx in range(n):
                d1 = int(d1_matrix[pred_idx, meas_idx] <= self.gate_matrix_thresh1)
                d2 = int(d2_matrix[pred_idx, meas_idx] >= self.gate_matrix_thresh2)
                gate_matrix[pred_idx, meas_idx] = d1 * d2 

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        obj_matches = set()
        det_matches = set() 
        unmatched_dets = [i for i in range(len(bboxes))] 
        # print("-------------")
        # print("GATE_MATRIX:")
        # print(gate_matrix)
        # print("-------------")
        for i in row_indices:
            for j in col_indices:
                # if self.objs[i].age > MAX_AGE:
                #     continue

                if gate_matrix[i,j] > 0:
                    obj_matches.add(i)
                    det_matches.add(j)
            if gate_matrix[i,col_indices].sum() > 0:
                unmatched_dets[j] = None
                 
        obj_matches = list(obj_matches)
        det_matches = list(det_matches)
        return obj_matches, det_matches, [ i for i in unmatched_dets if i != None ]


    def mahalanobis(self, predicted_bbox, bbox, obj):
        det_state = bbox_to_state(bbox) 
        state = bbox_to_state(predicted_bbox)
        S = obj.C.T @ obj.cov @ obj.C + obj.R 
        mh = ((det_state - state).T @ np.linalg.inv(S) @ (det_state - state))[0,0]
        return mh

    # this is good
    def cosine_distance(self, predicted_bbox, bbox):
        print(predicted_bbox.shape)
        patch1 = self.transform(predicted_bbox).unsqueeze(0)
        patch2 = self.transform(bbox).unsqueeze(0)
        
        patch1.requires_grad = True
        patch2.requires_grad = True
        output1, output2 = self.siamese_net(patch1, patch2)

        output1 = torch.flatten(output1)
        output2 = torch.flatten(output2)
        output1 = torch.unsqueeze(output1,0) #anc
        output2 = torch.unsqueeze(output2,0) #pos

        dist = F.cosine_similarity(x1=output1, x2=output2)
        return dist.item()

    @staticmethod
    def get_gaussian_mask():
        #128 is image size
        x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array([0.5,0.5])
        sigma = np.array([0.22,0.22])
        covariance = np.diag(sigma**2) 
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
        z = z.reshape(x.shape) 

        z = z / z.max()
        z  = z.astype(np.float32)
        cv2.imshow("gmask", z)
        mask = torch.from_numpy(z)

        return mask
