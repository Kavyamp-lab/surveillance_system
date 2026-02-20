import numpy as np
import math
from collections import deque
from config import Config


class BehaviorAnalyzer:
    def __init__(self):
        self.history = {}      # {id: deque[(cx, cy)]}
        self.last_seen = {}    # {id: frame_idx}

    def analyze(self, current_objects, frame_idx):
        behaviors = {}

        current_ids = set()

        # Extract centroids
        centroids = {}

        for obj in current_objects:
            x1, y1, x2, y2, tid = obj
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            centroids[tid] = (cx, cy)
            current_ids.add(tid)

        # ----------- Remove Dead IDs -----------
        dead_ids = set(self.history.keys()) - current_ids

        for tid in dead_ids:
            if tid in self.history:
                del self.history[tid]
            if tid in self.last_seen:
                del self.last_seen[tid]

        # ----------- Crowd Detection -----------
        in_crowd_ids = set()
        ids_list = list(centroids.keys())

        for i in range(len(ids_list)):
            for j in range(i + 1, len(ids_list)):
                id1 = ids_list[i]
                id2 = ids_list[j]

                c1 = centroids[id1]
                c2 = centroids[id2]

                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                if dist < Config.CROWD_DIST_THRESH:
                    in_crowd_ids.add(id1)
                    in_crowd_ids.add(id2)

        if len(in_crowd_ids) < Config.CROWD_MIN_PEOPLE:
            in_crowd_ids.clear()

        # ----------- Running & Loitering -----------
        for tid, (cx, cy) in centroids.items():

            self.last_seen[tid] = frame_idx

            if tid not in self.history:
                self.history[tid] = deque(maxlen=Config.HISTORY_MAX_FRAMES)

            self.history[tid].append((cx, cy))

            hist = self.history[tid]

            is_running = False
            is_loitering = False

            # Running detection
            if len(hist) > 5:
                px, py = hist[-5]
                dist = math.hypot(cx - px, cy - py)
                speed = dist / 5.0

                if speed > Config.RUNNING_SPEED_THRESH:
                    is_running = True

            # Loitering detection
            if len(hist) == Config.HISTORY_MAX_FRAMES:
                pts = np.array(hist)
                var_x = np.var(pts[:, 0])
                var_y = np.var(pts[:, 1])

                if var_x < Config.LOITERING_RADIUS and var_y < Config.LOITERING_RADIUS:
                    is_loitering = True

            behaviors[tid] = {
                "running": is_running,
                "loitering": is_loitering,
                "in_crowd": tid in in_crowd_ids
            }

        return behaviors