import cv2
import torch

from deepsort import DeepSort
from detector import YoloDetector

if __name__ == "__main__":
    detector = YoloDetector("yolov8n.pt") 
    device = torch.device("cpu")
    deepsort = DeepSort()
    cap = cv2.VideoCapture('test2.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if ret:
            bboxes = detector.eval(frame, filter_cls="car truck")
            for bbox in bboxes:
                deepsort.compare_with_last_frame(bbox, frame)

            deepsort.set_last_frame_bboxes(bboxes)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
