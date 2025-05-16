import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import json
from datetime import datetime

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get basic information about a video file."""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info

def save_analysis_results(results: Dict[str, Any], output_path: str = None) -> str:
    """Save analysis results to a JSON file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"analysis_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_path

def calculate_confidence_score(landmarks: List[Any]) -> float:
    """Calculate confidence score for pose detection."""
    if not landmarks:
        return 0.0
    
    # Get average visibility of all landmarks
    visibilities = [landmark.visibility for landmark in landmarks if hasattr(landmark, 'visibility')]
    return sum(visibilities) / len(visibilities) if visibilities else 0.0

def smooth_angles(angles: List[float], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to angle measurements."""
    if not angles:
        return []
    
    smoothed = []
    for i in range(len(angles)):
        start = max(0, i - window_size + 1)
        window = angles[start:i + 1]
        smoothed.append(sum(window) / len(window))
    
    return smoothed

def detect_posture_changes(angles: List[float], threshold: float) -> List[int]:
    """Detect points where posture changes significantly."""
    if len(angles) < 2:
        return []
    
    changes = []
    for i in range(1, len(angles)):
        if abs(angles[i] - angles[i-1]) > threshold:
            changes.append(i)
    
    return changes

def calculate_duration_stats(durations: List[float]) -> Dict[str, float]:
    """Calculate statistics for posture durations."""
    if not durations:
        return {
            'mean': 0.0,
            'max': 0.0,
            'min': 0.0,
            'std': 0.0
        }
    
    return {
        'mean': np.mean(durations),
        'max': max(durations),
        'min': min(durations),
        'std': np.std(durations)
    }

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"

def create_risk_level(analysis: Dict[str, Any]) -> str:
    """Calculate overall risk level based on analysis results."""
    risk_score = 0
    
    # Weight different factors
    if analysis['summary_flags']['trunk_bend_detected']:
        risk_score += 3
    if analysis['summary_flags']['arm_above_shoulder_detected']:
        risk_score += 2
    if analysis['summary_flags']['torso_twist_detected']:
        risk_score += 2
    
    # Check durations
    for category in ['neck', 'trunk', 'arms', 'legs']:
        for movement, info in analysis[category].items():
            if info['top_durations_secs'] and max(info['top_durations_secs']) > 10:
                risk_score += 1
    
    # Determine risk level
    if risk_score >= 5:
        return "High Risk"
    elif risk_score >= 3:
        return "Medium Risk"
    else:
        return "Low Risk" 