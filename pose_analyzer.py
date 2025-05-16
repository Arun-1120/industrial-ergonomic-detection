import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from collections import defaultdict

@dataclass
class PostureAnalysis:
    frame_number: int
    neck_forward_bend: bool
    neck_backward_bend: bool
    neck_side_bend: bool
    neck_twist: bool
    trunk_forward_bend: bool
    trunk_backward_bend: bool
    trunk_side_bend: bool
    trunk_twist: bool
    left_arm_raised: bool
    right_arm_raised: bool
    left_arm_behind: bool
    right_arm_behind: bool
    weight_on_left: bool
    weight_on_right: bool
    left_leg_bent: bool
    right_leg_bent: bool

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Reduced from 2 to 1 for faster processing
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Thresholds for posture detection
        self.NECK_FORWARD_THRESHOLD = 45
        self.NECK_BACKWARD_THRESHOLD = -20
        self.NECK_SIDE_THRESHOLD = 20
        self.TRUNK_FORWARD_THRESHOLD = 45
        self.TRUNK_BACKWARD_THRESHOLD = -20
        self.TRUNK_SIDE_THRESHOLD = 20
        self.ARM_RAISED_THRESHOLD = 90
        self.LEG_BEND_THRESHOLD = 120
        
        # Processing settings
        self.FRAME_SKIP = 2  # Process every 3rd frame
        self.MAX_WIDTH = 640  # Maximum width for processing

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle

    def analyze_frame(self, frame: np.ndarray) -> Tuple[PostureAnalysis, np.ndarray]:
        """Analyze a single frame for posture detection."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, frame
        
        # Draw pose landmarks
        annotated_frame = frame.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract key points
        nose = (landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[self.mp_pose.PoseLandmark.NOSE.value].y)
        left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        left_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y)
        right_hip = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        
        # Calculate angles
        neck_angle = self.calculate_angle(nose, left_shoulder, right_shoulder)
        trunk_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
        
        # Analyze postures
        analysis = PostureAnalysis(
            frame_number=0,  # Will be set by the video processor
            neck_forward_bend=neck_angle > self.NECK_FORWARD_THRESHOLD,
            neck_backward_bend=neck_angle < self.NECK_BACKWARD_THRESHOLD,
            neck_side_bend=abs(neck_angle - 90) > self.NECK_SIDE_THRESHOLD,
            neck_twist=False,  # Requires additional analysis
            trunk_forward_bend=trunk_angle > self.TRUNK_FORWARD_THRESHOLD,
            trunk_backward_bend=trunk_angle < self.TRUNK_BACKWARD_THRESHOLD,
            trunk_side_bend=abs(trunk_angle - 90) > self.TRUNK_SIDE_THRESHOLD,
            trunk_twist=False,  # Requires additional analysis
            left_arm_raised=self._is_arm_raised(landmarks, 'left'),
            right_arm_raised=self._is_arm_raised(landmarks, 'right'),
            left_arm_behind=self._is_arm_behind(landmarks, 'left'),
            right_arm_behind=self._is_arm_behind(landmarks, 'right'),
            weight_on_left=self._is_weight_on_leg(landmarks, 'left'),
            weight_on_right=self._is_weight_on_leg(landmarks, 'right'),
            left_leg_bent=self._is_leg_bent(landmarks, 'left'),
            right_leg_bent=self._is_leg_bent(landmarks, 'right')
        )
        
        return analysis, annotated_frame

    def _is_arm_raised(self, landmarks, side: str) -> bool:
        """Check if arm is raised above shoulder level."""
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value if side == 'left' 
                           else self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value if side == 'left'
                         else self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value if side == 'left'
                         else self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        angle = self.calculate_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y)
        )
        return angle > self.ARM_RAISED_THRESHOLD

    def _is_arm_behind(self, landmarks, side: str) -> bool:
        """Check if arm is pulled behind the body."""
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value if side == 'left'
                           else self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value if side == 'left'
                         else self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Check if wrist is behind the shoulder
        return wrist.x < shoulder.x if side == 'left' else wrist.x > shoulder.x

    def _is_weight_on_leg(self, landmarks, side: str) -> bool:
        """Check if weight is primarily on one leg."""
        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value if side == 'left'
                       else self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value if side == 'left'
                         else self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Simple check: if ankle is directly under hip
        return abs(hip.x - ankle.x) < 0.1

    def _is_leg_bent(self, landmarks, side: str) -> bool:
        """Check if leg is bent."""
        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value if side == 'left'
                       else self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value if side == 'left'
                        else self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value if side == 'left'
                         else self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        angle = self.calculate_angle(
            (hip.x, hip.y),
            (knee.x, knee.y),
            (ankle.x, ankle.y)
        )
        return angle < self.LEG_BEND_THRESHOLD

    def process_video(self, video_path: str) -> Dict:
        """Process entire video and return analysis results."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize counters and duration trackers
        posture_counts = defaultdict(int)
        current_durations = defaultdict(float)
        top_durations = defaultdict(list)
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate scaling factor if video is too large
        scale_factor = 1.0
        if width > self.MAX_WIDTH:
            scale_factor = self.MAX_WIDTH / width
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on FRAME_SKIP
            if frame_count % (self.FRAME_SKIP + 1) != 0:
                frame_count += 1
                continue
            
            # Resize frame if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            
            analysis, _ = self.analyze_frame(frame)
            if analysis:
                # Update counts and durations
                self._update_counts_and_durations(
                    analysis,
                    posture_counts,
                    current_durations,
                    top_durations,
                    (self.FRAME_SKIP + 1) / fps  # Adjust duration for skipped frames
                )
            
            frame_count += 1
            
        cap.release()
        
        # Prepare final report
        return self._prepare_report(
            frame_count,
            fps,
            posture_counts,
            top_durations
        )

    def _update_counts_and_durations(
        self,
        analysis: PostureAnalysis,
        counts: Dict,
        current_durations: Dict,
        top_durations: Dict,
        frame_duration: float
    ):
        """Update posture counts and durations."""
        # Helper function to update a single posture
        def update_posture(posture_name: str, is_active: bool):
            if is_active:
                counts[posture_name] += 1
                current_durations[posture_name] += frame_duration
            else:
                if current_durations[posture_name] > 0:
                    top_durations[posture_name].append(current_durations[posture_name])
                    top_durations[posture_name].sort(reverse=True)
                    top_durations[posture_name] = top_durations[posture_name][:3]
                    current_durations[posture_name] = 0

        # Update all postures
        update_posture('neck_forward_bending', analysis.neck_forward_bend)
        update_posture('neck_backward_bending', analysis.neck_backward_bend)
        update_posture('neck_side_bending', analysis.neck_side_bend)
        update_posture('neck_twisting', analysis.neck_twist)
        update_posture('trunk_forward_bending', analysis.trunk_forward_bend)
        update_posture('trunk_backward_bending', analysis.trunk_backward_bend)
        update_posture('trunk_side_bending', analysis.trunk_side_bend)
        update_posture('trunk_twisting', analysis.trunk_twist)
        update_posture('left_arm_above_shoulder', analysis.left_arm_raised)
        update_posture('right_arm_above_shoulder', analysis.right_arm_raised)
        update_posture('weight_on_left', analysis.weight_on_left)
        update_posture('weight_on_right', analysis.weight_on_right)

    def _prepare_report(
        self,
        total_frames: int,
        frame_rate: float,
        counts: Dict,
        top_durations: Dict
    ) -> Dict:
        """Prepare the final analysis report."""
        return {
            "total_frames": total_frames,
            "frame_rate": frame_rate,
            "neck": {
                "forward_bending": {
                    "count": counts.get('neck_forward_bending', 0),
                    "top_durations_secs": top_durations.get('neck_forward_bending', [])
                },
                "backward_bending": {
                    "count": counts.get('neck_backward_bending', 0),
                    "top_durations_secs": top_durations.get('neck_backward_bending', [])
                }
            },
            "trunk": {
                "forward_bending": {
                    "count": counts.get('trunk_forward_bending', 0),
                    "top_durations_secs": top_durations.get('trunk_forward_bending', [])
                },
                "twisting": {
                    "count": counts.get('trunk_twisting', 0),
                    "top_durations_secs": top_durations.get('trunk_twisting', [])
                }
            },
            "arms": {
                "left_arm_above_shoulder": {
                    "count": counts.get('left_arm_above_shoulder', 0),
                    "top_durations_secs": top_durations.get('left_arm_above_shoulder', [])
                },
                "right_arm_above_shoulder": {
                    "count": counts.get('right_arm_above_shoulder', 0),
                    "top_durations_secs": top_durations.get('right_arm_above_shoulder', [])
                }
            },
            "legs": {
                "weight_on_left": {
                    "count": counts.get('weight_on_left', 0),
                    "top_durations_secs": top_durations.get('weight_on_left', [])
                },
                "weight_on_right": {
                    "count": counts.get('weight_on_right', 0),
                    "top_durations_secs": top_durations.get('weight_on_right', [])
                }
            },
            "summary_flags": {
                "trunk_bend_detected": counts.get('trunk_forward_bending', 0) > 0,
                "arm_above_shoulder_detected": (
                    counts.get('left_arm_above_shoulder', 0) > 0 or
                    counts.get('right_arm_above_shoulder', 0) > 0
                ),
                "torso_twist_detected": counts.get('trunk_twisting', 0) > 0
            }
        } 