from config import Config

class RiskEngine:
    def __init__(self):
        self.risk_scores = {}  # {tid: score}

    def update_risks(self, behaviors, current_ids):

        # Remove IDs that disappeared
        dead_ids = set(self.risk_scores.keys()) - set(current_ids)
        for tid in dead_ids:
            del self.risk_scores[tid]

        # Update active IDs
        for tid in current_ids:

            if tid not in self.risk_scores:
                self.risk_scores[tid] = Config.RISK_BASELINE

            beh = behaviors.get(tid, {})
            score = self.risk_scores[tid]

            # Incremental penalties
            if beh.get("running"):
                score += Config.RISK_RUNNING_PENALTY

            if beh.get("loitering"):
                score += Config.RISK_LOITERING_PENALTY

            if beh.get("in_crowd"):
                score += Config.RISK_CROWD_PENALTY

            # Decay if normal behavior
            if not any(beh.values()):
                score -= Config.RISK_DECAY

            # Clamp bounds
            score = max(Config.RISK_BASELINE, min(100, score))

            self.risk_scores[tid] = score

        return {tid: self.risk_scores[tid] for tid in current_ids}