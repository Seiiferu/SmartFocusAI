# run_detector.py

import cv2
from src.detection.object_detector import ObjectDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = ObjectDetector(
        model_path="models/yolov5s.onnx",
        labels_path="models/coco.names",
        conf_threshold=0.3
    )

    while True:
        ret, frame = cap.read()
        if not ret: break

        dets = detector.detect(frame)
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,
                        f"{d['label']} {d['conf']*100:.0f}%",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
