# Trashformer

Trashformer is a proof-of-concept system that detects common waste items with YOLOv8, maps these YOLO COCO classes to the four bins on the robot (organic, paper/cardboard, plastics, landfill/other), and streams that information in real time for later use in the robot.

## Repo Layout

- `vision/`: all computer-vision work.
  - `src/categories.py`: maps YOLO COCO classes to trash categories and defines the `DetectionResult` data carrier (`TrashType` + metadata) used downstream.
  - `src/test_webcam.py`: Septekon webcam check without AI.
  - `src/test_yolo_img.py`: single-image YOLOv8 test.
  - `src/webcam_yolo_live.py`: runs YOLOv8 live on the webcam, shows annotated detections.
  - `src/trashformer_live.py`: full pipeline—YOLOv8 + trash categorization + JSON output for robot control.
  - `assets/images/test_image.jpg`: sample frame for `test_yolo_img.py`.
  - `src/yolov8n.pt`: default YOLOv8-nano weights.

## Prerequisites

- Python 3.9+ with `pip`.
- macOS with AVFoundation-compatible webcam (scripts open device index 0 using `cv2.CAP_AVFOUNDATION`).
- GPU would work better; YOLOv8n runs on CPU though for quick tests.

## Setup

1. Create and activate a virtual environment (recommended).

2. Install requirements:

   ```bash
   pip install -r vision/requirements
   ```

3. (Optional) Download different YOLOv8 weights via `yolo download` or the Ultralytics hub and place them next to `vision/src/yolov8n.pt`.

## Running the Examples

All scripts assume you run them from `vision/src` so relative imports work:

```bash
cd trashformer/vision/src
```

1. **Check the webcam (`test_webcam.py`)**

   ```bash
   python3 test_webcam.py
   ```

   Opens a raw feed to confirm the OS recognizes the Septekon camera. Press `q` to exit.

2. **YOLO test on a still image (`test_yolo_img.py`)**

   ```bash
   python3 test_yolo_img.py
   ```

   Loads `assets/images/test_image.jpg`, runs YOLOv8n, and displays an annotated image in a new window. Update `image_path` if you want to test other frames.

3. **Live YOLO detections (`webcam_yolo_live.py`)**

   ```bash
   python3 webcam_yolo_live.py
   ```

   Streams from the webcam, draws YOLO boxes, and filters detections below 0.5 confidence. Quit with `q`.

4. **Full Trashformer pipeline (`trashformer_live.py`)**

   ```bash
   python3 trashformer_live.py
   ```

   - Runs YOLOv8n every frame.
   - Converts each YOLO class into one of the four bins via `categorize_detection`.
   - Colors boxes by bin (plastics=blue, organics=green, paper=tan, landfill=gray) and labels them.
   - Prints a JSON list per frame with the trash class, confidence, raw YOLO class, bounding box, and image centroid so other subsystems can subscribe over a socket or shared queue.
   - Stop with `q`.

## Customizing Categories

- Extend `YOLO_TO_CATEGORY` in `vision/src/categories.py` with more COCO class names mapped to `TrashType` bins.
- If we need new/replaced bin types, add them to the `TrashType` enum and update the color switch in `trashformer_live.py`.

## Next Steps

- Use different YOLO weights (e.g., `yolov8s.pt`) if accuracy is more important than latency, but we need to ensure our hardware can keep up.
- Still need to pipe the JSON output from `trashformer_live.py` into a ROS node, serial link, or gRPC service that controls the arm and gripper.
- Stil need to populate `arm_kinematics/`, `gripper/`, and `mechanical_design/` with CAD, firmware, and control scripts so the README can reference full-stack setup later.
