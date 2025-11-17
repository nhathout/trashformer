# see if I can get yolov8 to output a bounding box
# & confidence level given a single frame

from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")

    image_path = "/Users/noahhathout/EK505/trashformer/vision/assets/images/test_image.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read image at {image_path}")
        return
    
    results = model(img)

    # get the annotated image
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 test image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()