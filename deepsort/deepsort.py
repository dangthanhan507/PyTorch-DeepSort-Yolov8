import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from torchvision import transforms as T


class DeepSort:
    def __init__(self) -> None:
        self.tracks = {} 
        self.last_frame_bboxes = None

        device = torch.device("cpu")
        self.siamese_net = torch.load("ckpts/model640.pt", map_location=device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((128, 128))
        ])

    def cascade_match(self):
        pass

    def compare_with_last_frame(self, bbox, frame):
        if not self.last_frame_bboxes:
            return
        else:
            for last_bbox in self.last_frame_bboxes:
                patch1 = frame[int(bbox.x0): int(bbox.x1), int(bbox.y0): int(bbox.y1)]
                patch1 = self.transform(patch1).unsqueeze(0)
                patch2 = frame[int(last_bbox.x0): int(last_bbox.x1), int(last_bbox.y0): int(last_bbox.y1)]
                patch2 = self.transform(patch2).unsqueeze(0)

                
                patch1.requires_grad = True
                patch2.requires_grad = True
                assert patch1.requires_grad == True and patch2.requires_grad == True
                output1, output2 = self.siamese_net(patch1, patch2)

                output1 = torch.flatten(output1)
                output2 = torch.flatten(output2)
                output1 = torch.unsqueeze(output1,0) #anc
                output2 = torch.unsqueeze(output2,0) #pos

                d1 = F.cosine_similarity(x1=output1, x2=output2)
                print(d1.item())

    def set_last_frame_bboxes(self,bboxes):
        self.last_frame_bboxes = bboxes

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
