import pytest
import numpy as np
import cv2
from pose_analyzer import PoseAnalyzer, PostureAnalysis

def test_calculate_angle():
    analyzer = PoseAnalyzer()
    
    # Test right angle
    angle = analyzer.calculate_angle((0, 0), (0, 1), (1, 1))
    assert abs(angle - 90) < 0.1
    
    # Test straight line
    angle = analyzer.calculate_angle((0, 0), (1, 1), (2, 2))
    assert abs(angle - 180) < 0.1
    
    # Test acute angle
    angle = analyzer.calculate_angle((0, 0), (1, 1), (2, 0))
    assert angle < 90

def test_analyze_frame_no_pose():
    analyzer = PoseAnalyzer()
    
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Analyze frame
    analysis, annotated_frame = analyzer.analyze_frame(frame)
    
    # Should return None for analysis when no pose is detected
    assert analysis is None
    assert annotated_frame.shape == frame.shape

def test_process_video_invalid_path():
    analyzer = PoseAnalyzer()
    
    # Test with non-existent video
    with pytest.raises(Exception):
        analyzer.process_video("nonexistent.mp4")

def test_posture_analysis_dataclass():
    # Test creating a PostureAnalysis instance
    analysis = PostureAnalysis(
        frame_number=1,
        neck_forward_bend=True,
        neck_backward_bend=False,
        neck_side_bend=False,
        neck_twist=False,
        trunk_forward_bend=True,
        trunk_backward_bend=False,
        trunk_side_bend=False,
        trunk_twist=False,
        left_arm_raised=True,
        right_arm_raised=False,
        left_arm_behind=False,
        right_arm_behind=False,
        weight_on_left=True,
        weight_on_right=False,
        left_leg_bent=False,
        right_leg_bent=False
    )
    
    # Verify values
    assert analysis.frame_number == 1
    assert analysis.neck_forward_bend is True
    assert analysis.neck_backward_bend is False
    assert analysis.trunk_forward_bend is True
    assert analysis.left_arm_raised is True
    assert analysis.weight_on_left is True

def test_arm_raised_detection():
    analyzer = PoseAnalyzer()
    
    # Create test landmarks with raised arm
    landmarks = {
        analyzer.mp_pose.PoseLandmark.LEFT_SHOULDER.value: type('Landmark', (), {'x': 0.5, 'y': 0.5})(),
        analyzer.mp_pose.PoseLandmark.LEFT_ELBOW.value: type('Landmark', (), {'x': 0.5, 'y': 0.3})(),
        analyzer.mp_pose.PoseLandmark.LEFT_WRIST.value: type('Landmark', (), {'x': 0.5, 'y': 0.1})()
    }
    
    # Test left arm raised
    assert analyzer._is_arm_raised(landmarks, 'left') is True
    
    # Test right arm (should be False as we only set left arm landmarks)
    assert analyzer._is_arm_raised(landmarks, 'right') is False

def test_leg_bent_detection():
    analyzer = PoseAnalyzer()
    
    # Create test landmarks with bent leg
    landmarks = {
        analyzer.mp_pose.PoseLandmark.LEFT_HIP.value: type('Landmark', (), {'x': 0.5, 'y': 0.5})(),
        analyzer.mp_pose.PoseLandmark.LEFT_KNEE.value: type('Landmark', (), {'x': 0.5, 'y': 0.7})(),
        analyzer.mp_pose.PoseLandmark.LEFT_ANKLE.value: type('Landmark', (), {'x': 0.5, 'y': 0.9})()
    }
    
    # Test left leg bent
    assert analyzer._is_leg_bent(landmarks, 'left') is True
    
    # Test right leg (should be False as we only set left leg landmarks)
    assert analyzer._is_leg_bent(landmarks, 'right') is False

def test_weight_distribution():
    analyzer = PoseAnalyzer()
    
    # Create test landmarks with weight on left leg
    landmarks = {
        analyzer.mp_pose.PoseLandmark.LEFT_HIP.value: type('Landmark', (), {'x': 0.5, 'y': 0.5})(),
        analyzer.mp_pose.PoseLandmark.LEFT_ANKLE.value: type('Landmark', (), {'x': 0.51, 'y': 0.9})(),
        analyzer.mp_pose.PoseLandmark.RIGHT_HIP.value: type('Landmark', (), {'x': 0.6, 'y': 0.5})(),
        analyzer.mp_pose.PoseLandmark.RIGHT_ANKLE.value: type('Landmark', (), {'x': 0.7, 'y': 0.9})()
    }
    
    # Test weight distribution
    assert analyzer._is_weight_on_leg(landmarks, 'left') is True
    assert analyzer._is_weight_on_leg(landmarks, 'right') is False 