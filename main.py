import cv2
import time
from config import Config
from tracker import ObjectTracker
from detector import SubDetector
from behavior import BehaviorAnalyzer
from risk_engine import RiskEngine
from alert_system import AlertEngine


def draw_hud(frame, obj, face_box, behaviors, risk_score):
    x1, y1, x2, y2, tid = obj

    # Base Box Color (Green -> Orange -> Red based on risk)
    if risk_score < 50:
        box_color = (0, 255, 0)
    elif risk_score < Config.RISK_ALERT_THRESHOLD:
        box_color = (0, 165, 255)
    else:
        box_color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Risk Meter (Above Box)
    meter_w = x2 - x1
    filled_w = int((risk_score / 100.0) * meter_w)

    cv2.rectangle(frame, (x1, y1 - 10), (x1 + meter_w, y1 - 5), (100, 100, 100), -1)
    cv2.rectangle(frame, (x1, y1 - 10), (x1 + filled_w, y1 - 5), box_color, -1)

    # ID & Risk Text
    label = f"ID: {tid} | R: {int(risk_score)}"
    cv2.putText(frame, label, (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Behavior Tags
    active_behs = [k for k, v in behaviors.items() if v]

    if active_behs:
        cv2.putText(frame, ",".join(active_behs),
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)

    # Face Bounding Box
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 1)


def main():
    print("Initializing Research-Grade Video Surveillance System...")

    tracker = ObjectTracker()
    sub_detector = SubDetector()
    behavior_analyzer = BehaviorAnalyzer()
    risk_engine = RiskEngine()
    alert_engine = AlertEngine()

    cap = cv2.VideoCapture(Config.VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: Could not open video source {Config.VIDEO_SOURCE}")
        return

    frame_idx = 0
    prev_time = time.time()

    print("System Online. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 1️⃣ Detection & Tracking
        objects = tracker.process_frame(frame)

        # 2️⃣ Behavior Analysis
        behaviors = behavior_analyzer.analyze(objects, frame_idx)

        # 3️⃣ Extract Track IDs
        current_ids = [obj[4] for obj in objects]

        # 4️⃣ Risk Engine Fusion
        risk_scores = risk_engine.update_risks(behaviors, current_ids)

        # 5️⃣ Alert Engine
        alert_engine.process(frame, objects, behaviors, risk_scores, frame_idx)

        # 6️⃣ Visualization
        for obj in objects:
            tid = obj[4]

            face_box = sub_detector.detect_face(frame, obj[:4])

            draw_hud(
                frame,
                obj,
                face_box,
                behaviors.get(tid, {}),
                risk_scores.get(tid, 0)
            )

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Intelligent Surveillance Interface", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System Shutdown. Analytics saved to system_logs.csv.")


if __name__ == "__main__":
    main()