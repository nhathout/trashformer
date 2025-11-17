# combining Webcam + YOLOv8 + categories.py

import json
import cv2
from ultralytics import YOLO

from categories import (
    categorize_detection,
    DetectionResult,
    TrashType,
)

def draw_detection(frame_bgr, det: DetectionResult, color):
    # draw bounding box (new colors) + label (trash type)
    x1, y1, x2, y2 = det.bbox_xyxy
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # draw the bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    # label with trash type + confidence
    label = f"{det.category.value} ({det.category_confidence:.2f})"

    # have to add another rectangle behind for the text
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame_bgr, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)

    # put the text
    cv2.putText(frame_bgr, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA,)

def main():
    # load the yolo model
    # can change the 'n' to s,m,l,x for bigger better model
    model = YOLO("yolov8n.pt")

    # open the webcam
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cam.isOpened():
        print("Can't open camera")
        return

    print("Webcam + YOLOv8 + 4-bin categorization. Press 'q' to quit.")

    while True:
        # get frame
        ret, frame = cam.read()
        if not ret:
            print("Can't get frame. Exiting ...")
            break

        # run yolo on the frame
        results = model(frame, conf=0.5, verbose=False)[0]

        detections_this_frame = []

        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy

                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]

                # get center of bbox
                cx = int((x1 + x2) / 2) # just avg of pts
                cy = int((y1 + y2) / 2)

                # map YOLO COCO class -> trash type
                det_res = categorize_detection(cls_name, conf)
                det_res.bbox_xyxy = (x1, y1, x2, y2)
                det_res.image_center = (cx, cy)

                detections_this_frame.append(det_res)

        # now we got all the detections from a single frame
        # we will then go through each detection and map them to a trash type
        for det in detections_this_frame:
            if det.category == TrashType.PLASTICS:
                color = (255, 0, 0) # blue-ish; can change later
            elif det.category == TrashType.ORGANIC:
                color = (0, 255, 0) # green-ish (bgr not rgb)
            elif det.category == TrashType.PAPER:
                color = (111, 167, 226) # yellow/brown-ish
            else: # landfill/other
                color = (128, 128, 128) # gray-ish

            draw_detection(frame, det, color=color)

        # display the annotated frame
        cv2.imshow("Trashformer Vision", frame)

        # print, send over socket, or make a json/csv for robot integration
        output_for_robot = [
            {
                "category": det.category.value,
                "category_confidence": det.category_confidence,
                "yolo_class": det.yolo_class,
                "yolo_confidence": det.yolo_confidence,
                "bbox_xyxy": det.bbox_xyxy,
                "image_center": det.image_center,
            }
            for det in detections_this_frame
        ]
        print(json.dumps(output_for_robot))

        # wait for 'q' to be pressed, then exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()