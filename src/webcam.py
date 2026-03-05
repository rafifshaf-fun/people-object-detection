def run_webcam():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = primary(frame)[0]
        for box in results.boxes:
            cls             = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi             = frame[y1:y2, x1:x2].copy()
            label           = "car" if cls == 0 else "person"
            box_color       = (0, 0, 255) if cls == 0 else (255, 165, 0)

            if cls == 0:
                for pb in plate_det(roi)[0].boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                    cv2.rectangle(roi, (px1, py1), (px2, py2), (0, 255, 0), 2)
            elif cls == 1:
                for fb in face_det(roi)[0].boxes:
                    fx1, fy1, fx2, fy2 = map(int, fb.xyxy[0])
                    cv2.rectangle(roi, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

            frame[y1:y2, x1:x2] = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.imshow("Surveillance Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()