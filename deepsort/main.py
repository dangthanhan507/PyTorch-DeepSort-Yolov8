import cv2
import torch

from deepsort import DeepSORT
from detector import YoloDetector


def is_okay_bbox(bbox):
    x0, x1 = int(bbox.x0), int(bbox.x1)
    y0, y1 = int(bbox.y0), int(bbox.y1)

    # if abs(x1 - x0) < 20 or abs(y1 - y0) < 20:
    #     return False
    # elif x0 > x1 or y0 > y1:
    #     return False
    return 0 <= x1 - x0 >= 20 and 0 <= y1 - y0 >= 20
    # return True


if __name__ == "__main__":
    detector = YoloDetector("yolov8n.pt") 
    device = torch.device("cpu")
    deepsort = DeepSORT()
    cap = cv2.VideoCapture('test3.mp4')
    # cap = cv2.VideoCapture("../mot_video.webm")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bboxes = detector.eval(frame)
            filtered_bboxes = list(filter(is_okay_bbox, bboxes))
            
            deepsort.track_boxes(filtered_bboxes, frame)
            frame = deepsort.drawTrack(frame)
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
