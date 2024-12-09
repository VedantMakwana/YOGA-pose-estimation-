# YOGA-pose-estimation
# Yoga Pose Estimation with MediaPipe and TensorFlow

## Project Overview

This project implements a real-time yoga pose estimation system using MediaPipe for pose detection and TensorFlow for pose classification. The application leverages computer vision techniques to identify and classify yoga poses in real-time through a webcam.

## Features

- Real-time pose detection using MediaPipe
- Yoga pose classification with a pre-trained TensorFlow model
- Confidence-based pose recognition
- Comprehensive body visibility checks
- Visual feedback for pose detection

## Prerequisites

### Hardware
- Webcam
- Computer with Python support

### Software
- Python 3.7+
- OpenCV
- MediaPipe
- TensorFlow
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yoga-pose-estimation.git
cd yoga-pose-estimation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
yoga-pose-estimation/
│
├── pose_estimation.py         # Main script for real-time pose detection
├── yoga_pose_classification_model.h5  # Pre-trained TensorFlow model
├── label_encoder.pkl           # Pose label encoder
├── requirements.txt            # Project dependencies
└── README.md                  # Project documentation
```

## Model Training

The project uses a pre-trained TensorFlow model for yoga pose classification. The model is trained on a dataset of yoga poses and saved as `yoga_pose_classification_model.h5`.

### Model Training Considerations
- Ensure diverse training data
- Include multiple variations of each pose
- Balance the dataset across different poses

## Usage

Run the pose estimation script:
```bash
python pose_estimation.py
```

### Interaction
- Press 'q' to quit the application
- The screen will display:
  - Detected pose name
  - Confidence score
  - Body landmark connections
  - Status messages for incomplete or uncertain detections

## Customization

### Confidence Thresholds
Adjust detection sensitivity in `real_time_pose_estimation()`:
- `pose_confidence_threshold`: Control landmark visibility
- `prediction_confidence_threshold`: Control pose classification confidence

## Troubleshooting

- Ensure proper lighting
- Position yourself fully in the camera frame
- Maintain a clear background
- Check camera permissions

## Performance Optimization

- Use a powerful GPU for faster inference
- Reduce model complexity if real-time performance is slow
- Optimize MediaPipe and TensorFlow settings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request


## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose detection
- [TensorFlow](https://www.tensorflow.org/) for machine learning framework
- [OpenCV](https://opencv.org/) for computer vision processing

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/yoga-pose-estimation](https://github.com/yourusername/yoga-pose-estimation)
