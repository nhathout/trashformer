# lets try to make yolov8n work live

import cv2
from ultralytics import YOLO

def main():
    # first load YOLOv8 model
    # yolov8n.pt (nano); later we can use yolov8s/m/l/x
    model = YOLO("yolov8n.pt")

    # open cam (Septekon)
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cam.isOpened():
        print("Cannot open camera")
        return
    
    print("Webcam + YOLOv8 live detection. Press 'q' to quit.")

    while True:
        # read a frame from the cam
        ret, frame = cam.read()
        if not ret:
            print("Can't get frame (cam disconnected?). Exiting ...")
            break

        # run yolo on the frame
        # conf = 0.5 igores all detections under 50% confidence
        results = model(frame, conf=0.5, verbose=False)[0]

        # get annotated frame
        annotated_frame = results.plot()

        cv2.imshow("YOLOv8 Live - Webcam", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cam.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()