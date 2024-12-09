import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

def load_model_and_encoder(model_path, encoder_path):
    model = load_model(model_path)
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return model, label_encoder

def preprocess_landmarks(landmarks):
    keypoints = []
    for lm in landmarks:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return np.array(keypoints).reshape(1, -1)

def check_body_visibility(landmarks):
    # Define key body part landmark indices from MediaPipe Pose
    upper_body_landmarks = [
        11,  # Left shoulder
        12,  # Right shoulder
        13,  # Left elbow
        14,  # Right elbow
        15,  # Left wrist
        16,  # Right wrist
    ]
    
    lower_body_landmarks = [
        23,  # Left hip
        24,  # Right hip
        25,  # Left knee
        26,  # Right knee
        27,  # Left ankle
        28,  # Right ankle
    ]
    
    # Visibility threshold
    visibility_threshold = 0.5
    
    # Check upper body visibility
    upper_body_visible = sum(
        1 for idx in upper_body_landmarks 
        if landmarks[idx].visibility > visibility_threshold
    )
    
    # Check lower body visibility
    lower_body_visible = sum(
        1 for idx in lower_body_landmarks 
        if landmarks[idx].visibility > visibility_threshold
    )
    
    # Require at least 50% of landmarks in both upper and lower body to be visible
    upper_body_check = upper_body_visible >= len(upper_body_landmarks) // 2
    lower_body_check = lower_body_visible >= len(lower_body_landmarks) // 2
    
    return upper_body_check and lower_body_check

def real_time_pose_estimation(model, label_encoder, 
                               pose_confidence_threshold=0.7,
                               prediction_confidence_threshold=0.6):

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image with MediaPipe Pose
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Enhanced pose detection with body visibility check
        if results.pose_landmarks:
            # Check body visibility
            if check_body_visibility(results.pose_landmarks.landmark):
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Preprocess the landmarks
                keypoints = preprocess_landmarks(results.pose_landmarks.landmark)

                # Predict the pose with confidence check
                predictions = model.predict(keypoints)
                max_prediction = np.max(predictions)
                predicted_index = np.argmax(predictions)

                # Only display pose if prediction confidence is high enough
                if max_prediction >= prediction_confidence_threshold:
                    predicted_pose = label_encoder.inverse_transform([predicted_index])[0]
                    cv2.putText(image, f"Pose: {predicted_pose} ({max_prediction:.2f})", 
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Pose: Uncertain", 
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Incomplete Body View", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Yoga Pose Estimation', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    # Paths to the model and encoder
    model_path = 'yoga_pose_classification_model.h5'  # Replace with your model path
    encoder_path = 'label_encoder.pkl'  # Replace with your encoder path

    # Load the model and LabelEncoder
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)

    # Run real-time yoga pose estimation
    real_time_pose_estimation(model, label_encoder)