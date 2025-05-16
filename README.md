# Industrial Ergonomic Risk Detection

An AI-powered application for detecting and analyzing ergonomic risks in industrial work environments using pose estimation.

## Features

- Real-time pose estimation using MediaPipe
- Analysis of various body movements:
  - Neck movements (forward/backward/side bending)
  - Trunk movements (forward/backward/side bending, twisting)
  - Arm positions (above shoulder, behind back)
  - Leg weight distribution
- Risk assessment and scoring
- AI-generated recommendations
- Visual analysis with charts and graphs
- Export results to JSON format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/industrial-ergonomic-detection.git
cd industrial-ergonomic-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a video file for analysis

4. View the results and download the analysis report

## Project Structure

```
industrial-ergonomic-detection/
├── app.py                 # Main Streamlit application
├── pose_analyzer.py       # Pose analysis logic
├── requirements.txt       # Project dependencies
├── tests/                 # Test cases
│   └── test_pose_analyzer.py
└── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- Streamlit
- Plotly
- Pandas

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose estimation
- Streamlit for the web interface
- Plotly for data visualization