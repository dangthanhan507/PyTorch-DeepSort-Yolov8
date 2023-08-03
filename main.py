import cv2
from detector import YoloDetector
from sort import SORT

if __name__ == '__main__':
    cap = cv2.VideoCapture('mot_easy.webm')

    yolo = YoloDetector('yolov8m.pt')
    sort = SORT()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bboxes = yolo.eval(frame)
            sort.track_boxes(bboxes)
            # for bbox in bboxes:
            #     bbox.drawBox(frame)
            frame = sort.drawTrack(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
        