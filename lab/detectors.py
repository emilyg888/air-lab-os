# lab/detectors.py

from core.pattern import Pattern


# ---------------------------
# Helper functions
# ---------------------------

def avg(values):
    return sum(values) / len(values) if values else 0


# ---------------------------
# Detector 1: Spike Detector
# ---------------------------

def spike_detector(threshold=2.0):
    def logic(data):
        values = data["transactions"]
        mean = avg(values)

        spikes = [v for v in values if v > mean * threshold]

        return {
            "detector": "spike",
            "spike_count": len(spikes),
            "score": len(spikes)
        }

    return Pattern("spike_detector", logic)


# ---------------------------
# Detector 2: Velocity Detector
# ---------------------------

def velocity_detector(threshold=3):
    def logic(data):
        values = data["transactions"]

        high_freq = sum(1 for v in values if v > 0)

        flag = high_freq > threshold

        return {
            "detector": "velocity",
            "high_freq_count": high_freq,
            "score": high_freq if flag else 0
        }

    return Pattern("velocity_detector", logic)


# ---------------------------
# Detector 3: Simple Z-score Detector
# ---------------------------

def zscore_detector(z_threshold=2.0):
    def logic(data):
        values = data["transactions"]
        mean = avg(values)
        variance = avg([(v - mean) ** 2 for v in values])
        std = variance ** 0.5

        anomalies = [
            v for v in values
            if std > 0 and abs((v - mean) / std) > z_threshold
        ]

        return {
            "detector": "zscore",
            "anomaly_count": len(anomalies),
            "score": len(anomalies)
        }

    return Pattern("zscore_detector", logic)


# ---------------------------
# Registry of detectors (lab mode)
# ---------------------------

def get_detectors():
    return [
        spike_detector(),
        velocity_detector(),
        zscore_detector()
    ]