import cv2
import numpy as np

def save_contoured_frame(video_path, frame_number, threshold=50, save_path='/path/to/save/contoured_frame.png'):
    """
    Save a frame from the video with contours drawn on it.
    
    Parameters:
    video_path (str): Path to the video file.
    frame_number (int): The number of the frame to process.
    threshold (int): Threshold value for contour detection.
    save_path (str): Path to save the contoured frame image.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter == frame_number:
            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the frame
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 4)

            # Save the contoured frame
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            print(largest_contour)
            print(x, y, w, h)
            x, y, w, h = 739, 339, 408, 408
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(save_path, frame)
            break

        frame_counter += 1

    cap.release()

# Example usage
save_contoured_frame('./images/2023-11-26 21-34-05.mkv', 250, save_path='contoured_frame.png')
