# make sure the Septekon camera works

import cv2

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cam.isOpened():
        print("Cannot open camera")
        return
    
    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't get frame (cam disconnected?). Exiting ...")
            break

        # Display captured frame
        cv2.imshow("Webcam Test", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()