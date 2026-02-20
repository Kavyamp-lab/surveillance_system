import os
import cv2
import csv
from datetime import datetime
from config import Config
import winsound

class AlertEngine:
    def __init__(self):
        self.alert_dir = "alerts"
        self.log_file = "system_logs.csv"
        os.makedirs(self.alert_dir, exist_ok=True)

        self.cooldowns = {}  # {track_id: last_alert_frame}

        # Create CSV with header if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_index",
                    "track_id",
                    "risk_score",
                    "behaviors"
                ])

    def process(self, frame, objects, behaviors, risk_scores, frame_idx):
        logs = []

        for obj in objects:
            x1, y1, x2, y2, tid = obj

            score = risk_scores.get(tid, 0)
            beh_dict = behaviors.get(tid, {})

            # Extract active behaviors
            active_behs = [k for k, v in beh_dict.items() if v]

            # -------- Trigger Alert --------
            if score >= Config.RISK_ALERT_THRESHOLD:

                last_alert_frame = self.cooldowns.get(tid, -1000)

                # 60 frame cooldown
                if frame_idx - last_alert_frame > 60:
                    self.cooldowns[tid] = frame_idx

                    self.save_snapshot(
                        frame,
                        tid,
                        score,
                        active_behs
                    )

            # -------- Logging --------
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logs.append([
                timestamp,
                frame_idx,
                tid,
                score,
                ",".join(active_behs) if active_behs else "normal"
            ])

        # Write logs
        if logs:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(logs)

    def save_snapshot(self, frame, tid, score, behaviors):
        timestamp = datetime.now().strftime("%H%M%S")

        beh_str = "_".join(behaviors) if behaviors else "suspicious"

        filename = f"{self.alert_dir}/alert_ID{tid}_{timestamp}_risk{int(score)}_{beh_str}.jpg"

        cv2.imwrite(filename, frame)

        # 🔔 Play sound (non-blocking)
        try:
            winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
        except:
            winsound.Beep(1500, 500)  # fallback beep

        print(f"[ALERT] High risk detected for ID {tid}. Snapshot saved: {filename}")