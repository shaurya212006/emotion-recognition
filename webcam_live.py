import cv2, time, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import process_image

print("EmoScan Live — Press Q to quit, S to save frame")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps, frame_count, t_start = 0, 0, time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    annotated, results = process_image(frame)

    frame_count += 1
    elapsed = time.time() - t_start
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        t_start = time.time()

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    cv2.imshow("EmoScan Live", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s'):
        fname = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"Saved: {fname}")

cap.release()
cv2.destroyAllWindows()