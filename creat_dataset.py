import os
import cv2
import mediapipe as mp 
import matplotlib.pyplot as plt
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True,min_detection_confidence =0.3,model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def extract_pose_data(image_path, pose):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)
    
    # If pose landmarks are found, extract their coordinates
    if results.pose_landmarks:
        pose_data = {}
        for landmark in mp_pose.PoseLandmark:
            # Get the corresponding landmark coordinates
            landmark_data = results.pose_landmarks.landmark[landmark.value]
            
            # Store the coordinates in pose_data with dynamic names based on the landmark name
            pose_data[f"{landmark.name}_x"] = landmark_data.x
            pose_data[f"{landmark.name}_y"] = landmark_data.y
        
        return pose_data
    else:
        return None


# Function to process the folder and subfolders
def process_yoga_poses(folder_path):
    # Initialize the dataframe with the pose points and a Pose_name column
    df = pd.DataFrame()  # Store extracted data

    # Loop through each subfolder (yoga pose folder)
    for pose_folder in os.listdir(folder_path):
        pose_folder_path = os.path.join(folder_path, pose_folder)
        
        if os.path.isdir(pose_folder_path):  # Only process subfolders
            for file_name in os.listdir(pose_folder_path):
                file_path = os.path.join(pose_folder_path, file_name)
                
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')): 
                    # Process image files
                    print(f"Processing: {file_name}")
                    pose_data = extract_pose_data(file_path, pose)
                    
                    if pose_data:
                        pose_data["Pose_name"] = pose_folder  # Add pose name
                        # Convert pose data to a DataFrame and concatenate with the main DataFrame
                        pose_df = pd.DataFrame([pose_data])
                        df = pd.concat([df, pose_df], ignore_index=True)
    
    return df

# Usage Example
folder_path = "yoga_pose"
processed_data = process_yoga_poses(folder_path)

# Save the DataFrame to a CSV
processed_data.to_csv("pose_data.csv", index=False)