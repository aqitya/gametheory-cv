import cv2
import numpy as np
from PIL import Image

# Define the color ranges for red and blue pieces
# These ranges may need to be adjusted based on the specific colors in the image
red_lower = (150, 0, 0)
red_upper = (255, 100, 100)
blue_lower = (0, 0, 150)
blue_upper = (100, 100, 255)


# Function to convert the board array to a string representation
def board_to_string_with_turn(board_array):
    string_repr = "R_A_4_4_----"
    for row in board_array:
        for cell in row:
            if cell == 1:
                string_repr += 'x' # Blue
            elif cell == -1:
                string_repr += 'o' # Red
            else:
                string_repr += '-' # Empty
        string_repr += '-'  # Add dead character after each row
    return string_repr


# Function to check if a pixel is within the color range
def is_color(pixel, lower, upper):
    return all(lower[i] <= pixel[i] <= upper[i] for i in range(3))

def process_frame(frame):
    # Convert the frame to a PIL Image and then to RGB
    # Initialize an empty 2D array for the board representation
    x, y, w, h = find_board_bounds(frame)

    # Crop the frame to the board area
    cropped_frame = frame[y:y+h, x:x+w]

    # Convert the cropped frame to a PIL Image and then to RGB
    img = Image.fromarray(cropped_frame).convert('RGB')
    board_size = (3, 3)  # Assuming a 3x3 Dodgem board
    board = np.full(board_size, 0, dtype=int)
    # Process the image to fill the board array
    width, height = img.size
    cell_width = width // board_size[1]
    cell_height = height // board_size[0]

    for i in range(board_size[0]):
        for j in range(board_size[1]):
            # Calculate the center of each cell
            center_x = j * cell_width + cell_width // 2
            center_y = i * cell_height + cell_height // 2
            pixel = img.getpixel((center_x, center_y))
            
            # checks only red and blue, thus cursor being in the middle is fine.
            if is_color(pixel, red_lower, red_upper):
                board[i, j] = -1
            elif is_color(pixel, blue_lower, blue_upper):
                board[i, j] = 1

    return board

# determines if there is motion in the frame
def is_motion(previous_frame, current_frame, threshold=100000):
    frame_delta = cv2.absdiff(previous_frame, current_frame)
    thresholded = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_level = np.sum(thresholded)
    return motion_level > threshold

# Function to find the bounds of the board in the frame
# Useless for now, but will be useful when combining with the CV image processing steps from Connect4 CV
def find_board_bounds(frame, threshold=50):
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    '''
    Hard-coding the location of the corners of the video. However, this can be done using the Connect4 CV image processing steps.
    '''
    if contours:
        x, y, w, h = 739, 339, 408, 408
        return x, y, w, h
    else:
        return 0, 0, frame.shape[1], frame.shape[0] # Return the entire frame if no contours are found
    

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the previous position string for comparison
    previous_position_string = None
    turn_char = 'A'

    # Read the first frame
    ret, previous_frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from video.")
        cap.release()
        return
    
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Check for motion
        if not is_motion(previous_frame, gray):
            board_array = process_frame(frame)
            current_position_string = board_to_string_with_turn(board_array)
            
            # Check for change in position string
            if previous_position_string and current_position_string[-16:] != previous_position_string[-16:]:
                turn_char = 'B' if turn_char == 'A' else 'A'
                current_position_string = f"R_{turn_char}_4_4_----" + current_position_string[-12:]
                print(f"{current_position_string}")
            
                # Check if the game is over
                if current_position_string[-16:].count('x') == 1 and current_position_string[-16:].count('o') == 0:
                    print("Game Over detected. Player 1 won the game!")
                    break
                elif current_position_string[-16:].count('x') == 0 and current_position_string[-16:].count('o') == 1:
                    print("Game Over detected. Player 1 lost the game!")
                    break

            # Update the previous position string for the next iteration
            previous_position_string = current_position_string

        previous_frame = gray

    cap.release()

# Calling the video
video_path = './images/2023-11-26 21-34-05.mkv'
extract_frames(video_path)