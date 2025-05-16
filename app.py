import streamlit as st
import cv2
import tempfile
import os
import json
from pose_analyzer import PoseAnalyzer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import time
from PIL import Image
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Body part icons using emojis
BODY_PART_ICONS = {
    "neck": "👤",  # Using head emoji for neck
    "trunk": "👕",  # Using t-shirt emoji for trunk
    "arms": "💪",  # Using flexed biceps emoji for arms
    "legs": "🦵"   # Using leg emoji for legs
}

# Action icons
ACTION_ICONS = {
    "forward_bending": "⬇️",
    "backward_bending": "⬆️",
    "side_bending": "↔️",
    "twisting": "🔄",
    "arm_raised": "✋",
    "arm_behind": "🤲",
    "weight_shift": "⚖️"
}

def get_icon_html(icon_url: str, size: int = 50) -> str:
    """Generate HTML for an icon with specified size."""
    return f'<img src="{icon_url}" width="{size}" height="{size}">'

def display_body_part_section(category: str, results: Dict[str, Any], st_container):
    """Display analysis for a specific body part with icons."""
    st_container.markdown(f"### {BODY_PART_ICONS[category]} {category.title()} Analysis")
    
    # Display movements with icons
    for movement, info in results[category].items():
        if info['count'] > 0:
            icon = ACTION_ICONS.get(movement.split('_')[0], '')
            st_container.markdown(f"- {icon} {movement.replace('_', ' ').title()}: {info['count']} times")
            if info['top_durations_secs']:
                st_container.markdown(f"  - Longest duration: {max(info['top_durations_secs']):.1f} seconds")

def generate_analysis_comments(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI-based analysis comments and recommendations."""
    comments = {
        "risk_assessment": "",
        "key_findings": [],
        "recommendations": []
    }
    
    # Risk Assessment
    risk_score = 0
    if results['summary_flags']['trunk_bend_detected']:
        risk_score += 3
    if results['summary_flags']['arm_above_shoulder_detected']:
        risk_score += 2
    if results['summary_flags']['torso_twist_detected']:
        risk_score += 2
    
    # Check durations for prolonged postures
    for category in ['neck', 'trunk', 'arms', 'legs']:
        for movement, info in results[category].items():
            if info['top_durations_secs'] and max(info['top_durations_secs']) > 10:
                risk_score += 1
    
    if risk_score >= 5:
        comments["risk_assessment"] = "⚠️ High Risk: Multiple ergonomic risk factors detected with prolonged durations."
    elif risk_score >= 3:
        comments["risk_assessment"] = "⚡ Medium Risk: Some ergonomic concerns identified."
    else:
        comments["risk_assessment"] = "✅ Low Risk: Minimal ergonomic concerns detected."
    
    # Key Findings with icons
    if results['trunk']['forward_bending']['count'] > 0:
        comments["key_findings"].append(
            f"{ACTION_ICONS['forward_bending']} Trunk forward bending detected {results['trunk']['forward_bending']['count']} times, "
            f"with longest duration of {max(results['trunk']['forward_bending']['top_durations_secs']):.1f} seconds"
        )
    
    if results['arms']['left_arm_above_shoulder']['count'] > 0 or results['arms']['right_arm_above_shoulder']['count'] > 0:
        comments["key_findings"].append(
            f"{ACTION_ICONS['arm_raised']} Arms raised above shoulder level detected {results['arms']['left_arm_above_shoulder']['count'] + results['arms']['right_arm_above_shoulder']['count']} times"
        )
    
    if results['neck']['forward_bending']['count'] > 0:
        comments["key_findings"].append(
            f"{ACTION_ICONS['forward_bending']} Neck forward bending detected {results['neck']['forward_bending']['count']} times, "
            f"with longest duration of {max(results['neck']['forward_bending']['top_durations_secs']):.1f} seconds"
        )
    
    # Recommendations with icons
    if results['trunk']['forward_bending']['count'] > 0:
        comments["recommendations"].append(
            f"🔄 Consider using lifting equipment or adjusting work height to reduce trunk bending"
        )
    
    if results['arms']['left_arm_above_shoulder']['count'] > 0 or results['arms']['right_arm_above_shoulder']['count'] > 0:
        comments["recommendations"].append(
            f"📏 Adjust work height or use tools to reduce overhead work"
        )
    
    if results['neck']['forward_bending']['count'] > 0:
        comments["recommendations"].append(
            f"💺 Adjust monitor height and take regular breaks to reduce neck strain"
        )
    
    if results['legs']['weight_on_left']['count'] > results['legs']['weight_on_right']['count'] * 1.5:
        comments["recommendations"].append(
            f"⚖️ Consider alternating weight distribution between legs to reduce strain"
        )
    
    return comments

def create_duration_chart(data: Dict[str, Any], category: str) -> go.Figure:
    """Create a bar chart for duration data."""
    durations = []
    labels = []
    
    for movement, info in data[category].items():
        for duration in info['top_durations_secs']:
            durations.append(duration)
            labels.append(f"{movement.replace('_', ' ').title()}")
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=durations)
    ])
    
    fig.update_layout(
        title=f"Top Duration of {category.title()} Movements",
        xaxis_title="Movement Type",
        yaxis_title="Duration (seconds)",
        showlegend=False
    )
    
    return fig

def create_count_chart(data: Dict[str, Any], category: str) -> go.Figure:
    """Create a bar chart for count data."""
    counts = []
    labels = []
    
    for movement, info in data[category].items():
        counts.append(info['count'])
        labels.append(f"{movement.replace('_', ' ').title()}")
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=counts)
    ])
    
    fig.update_layout(
        title=f"Count of {category.title()} Movements",
        xaxis_title="Movement Type",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def main():
    try:
        # Set page config
        st.set_page_config(
            page_title="Industrial Ergonomic Risk Detection",
            page_icon="🏭",
            layout="wide"
        )
        
        st.title("🏭 Industrial Ergonomic Risk Detection")
        st.markdown("""
        This application analyzes videos of industrial work to detect potentially risky postures.
        Upload a video to get started with the analysis.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            try:
                # Create a temporary file to store the uploaded video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                # Get video duration
                duration = get_video_duration(video_path)
                st.info(f"Video duration: {duration:.1f} seconds")
                
                # Initialize pose analyzer
                analyzer = PoseAnalyzer()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                start_time = time.time()
                results = analyzer.process_video(video_path)
                end_time = time.time()
                
                # Clean up temporary file
                os.unlink(video_path)
                
                # Show processing time
                processing_time = end_time - start_time
                st.success(f"Analysis completed in {processing_time:.1f} seconds")
                
                # Generate AI analysis comments
                analysis_comments = generate_analysis_comments(results)
                
                # Display results
                st.header("Analysis Results")
                
                # AI Analysis Section
                st.subheader("🤖 AI Analysis")
                
                # Risk Assessment
                st.markdown(f"### Risk Level: {analysis_comments['risk_assessment']}")
                
                # Key Findings
                st.markdown("### Key Findings")
                for finding in analysis_comments['key_findings']:
                    st.markdown(f"- {finding}")
                
                # Recommendations
                st.markdown("### Recommendations")
                for recommendation in analysis_comments['recommendations']:
                    st.markdown(f"- {recommendation}")
                
                # Body Part Analysis
                st.subheader("Body Part Analysis")
                
                # Create tabs for different categories
                tab1, tab2, tab3, tab4 = st.tabs(["Neck", "Trunk", "Arms", "Legs"])
                
                with tab1:
                    display_body_part_section('neck', results, st)
                    st.plotly_chart(create_count_chart(results, 'neck'), use_container_width=True)
                    st.plotly_chart(create_duration_chart(results, 'neck'), use_container_width=True)
                
                with tab2:
                    display_body_part_section('trunk', results, st)
                    st.plotly_chart(create_count_chart(results, 'trunk'), use_container_width=True)
                    st.plotly_chart(create_duration_chart(results, 'trunk'), use_container_width=True)
                
                with tab3:
                    display_body_part_section('arms', results, st)
                    st.plotly_chart(create_count_chart(results, 'arms'), use_container_width=True)
                    st.plotly_chart(create_duration_chart(results, 'arms'), use_container_width=True)
                
                with tab4:
                    display_body_part_section('legs', results, st)
                    st.plotly_chart(create_count_chart(results, 'legs'), use_container_width=True)
                    st.plotly_chart(create_duration_chart(results, 'legs'), use_container_width=True)
                
                # Raw data
                st.subheader("Raw Data")
                st.json(results)
                
                # Download button for results
                st.download_button(
                    label="Download Analysis Results",
                    data=json.dumps({
                        "analysis_results": results,
                        "ai_analysis": analysis_comments
                    }, indent=2),
                    file_name="ergonomic_analysis.json",
                    mime="application/json"
                )
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                st.error(f"An error occurred while processing the video: {str(e)}")
                if 'video_path' in locals():
                    try:
                        os.unlink(video_path)
                    except:
                        pass
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred in the application. Please try again.")

if __name__ == "__main__":
    main() 