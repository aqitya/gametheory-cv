import numpy as np
import cv2
from matplotlib.pyplot import imread

# Load the mask into a NumPy array.
mask = imread('/Users/aditummala/Desktop/gametheory/homefuns/image.png')

# Check if the image has more than one channel (e.g., RGB or RGBA)
if mask.ndim > 2:
    # Convert the image to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

# Apply thresholding to ensure the mask is binary
_, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# Check the unique values in the mask_binary
unique_values = np.unique(mask_binary)
print("Unique values in the binary mask:", unique_values)

# If there are more than two unique values, reapply thresholding
if len(unique_values) > 2:
    _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(
    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Define the dimensions of the Connect 4 grid
rows = 6
columns = 7

# Initialize the game grid with empty slots
grid = np.zeros((rows, columns), dtype=int)

# Calculate the height and width of each slot
slot_height = mask_binary.shape[0] // rows
slot_width = mask_binary.shape[1] // columns

# Loop through each contour found in the mask
for cnt in contours:
    # Get the bounding rectangle around the contour
    y, x, _, _ = cv2.boundingRect(cnt)

    # Determine the row and column in the grid representation
    row = rows - 1 - int(y // slot_height)
    col = int(x // slot_width)

    # Update the grid
    grid[row, col] = 1

# Print the resulting grid
print(grid)
