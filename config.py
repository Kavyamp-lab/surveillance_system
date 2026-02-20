# config.py

class Config:
    # Model & Input
    YOLO_MODEL = "yolov8n.pt"
    VIDEO_SOURCE = 0  # 0 for webcam, or path to "video.mp4"
    CONF_THRESHOLD = 0.4
    
    # Tracking & Temporal Buffer
    HISTORY_MAX_FRAMES = 30     # Memory buffer size per person
    CLEANUP_AGE = 50            # Frames after which inactive IDs are purged
    
    # Behavior Analysis Thresholds
    RUNNING_SPEED_THRESH = 20.0 # Pixels per frame
    LOITERING_TIME_THRESH = 90  # Frames stayed in same area (~3 secs at 30fps)
    LOITERING_RADIUS = 30.0     # Maximum movement variance to be considered loitering
    CROWD_DIST_THRESH = 120.0   # Pairwise distance to group a crowd (pixels)
    CROWD_MIN_PEOPLE = 3        # Minimum people to trigger a crowd alert
    
    # Risk Engine (0-100 Scale)
    RISK_BASELINE = 10
    RISK_RUNNING_PENALTY = 15
    RISK_LOITERING_PENALTY = 5
    RISK_CROWD_PENALTY = 10
    RISK_DECAY = 1              # Decay rate per frame if normal
    RISK_ALERT_THRESHOLD = 75   # Threshold to trigger an alert