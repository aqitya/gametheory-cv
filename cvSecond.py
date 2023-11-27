import cv2
import numpy as np
from colorthief import ColorThief


def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)

img = cv2.imread('./result_image2.png')

# Constants
new_width = 500
img_h, img_w, _ = img.shape
scale = new_width / img_w
img_w = int(img_w * scale)
img_h = int(img_h * scale)
img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
img_orig = img.copy()
imgshow('Original Image (Resized)', img_orig)

# Bilateral Filter
bilateral_filtered_image = cv2.bilateralFilter(img, 15, 190, 190)
imgshow('Bilateral Filter', bilateral_filtered_image)

# Calculate the size of each grid cell
cell_width = img_w // 7
cell_height = img_h // 6

# Create a copy of the original image to draw the grid
grid_image = img_orig.copy()

# Draw vertical lines for the grid
for i in range(1, 7):
    cv2.line(grid_image, (i * cell_width, 0), (i * cell_width, img_h), (0, 255, 0), 1)

# Draw horizontal lines for the grid
for i in range(1, 6):
    cv2.line(grid_image, (0, i * cell_height), (img_w, i * cell_height), (0, 255, 0), 1)

# Display the image with the grid
cv2.imwrite('grid_image_with_cells.png', grid_image)
imgshow('Grid Image', grid_image)


# Initialize the grid
grid = np.zeros((6, 7))

RED_LOWER_HSV = np.array([0, 100, 100])
RED_UPPER_HSV = np.array([10, 255, 255])
BLACK_LOWER_HSV = np.array([0, 0, 0])
BLACK_UPPER_HSV = np.array([180, 255, 50])

# Function to check if the majority of the pixels in a masked area are of the color of the mask
def is_color_dominant(mask):
    # Count the non-zero (white) pixels in the mask
    white_pixels = cv2.countNonZero(mask)
    # Calculate the percentage of white pixels
    white_area_ratio = white_pixels / mask.size
    # If the white area covers more than 20% of the mask, we consider the color to be dominant
    return white_area_ratio > 0.2

def process_cell(img, x_start, y_start, width, height):
    # Crop the cell from the image
    cell_img = img[y_start:y_start + height, x_start:x_start + width]

    # Convert the cell image to grayscale
    gray_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    # Apply Hough Circle Transform to find circles in the grayscale image
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Ensure the crop coordinates are within the image bounds
            x, y, r = int(x), int(y), int(r)
            x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(width, x + r), min(height, y + r)

            # Crop the circle from the cell_img
            circle_img = cell_img[y1:y2, x1:x2]

            # If the circle_img is empty, skip to the next circle
            if circle_img.size == 0:
                continue

            # Convert to HSV and create masks for red and black
            hsv_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv_circle_img, RED_LOWER_HSV, RED_UPPER_HSV)
            black_mask = cv2.inRange(hsv_circle_img, BLACK_LOWER_HSV, BLACK_UPPER_HSV)

            if is_color_dominant(red_mask):
                return 1  # Red 
            elif is_color_dominant(black_mask):
                return -1  # Black 

    # No circles detected or not the right color
    return 0

# Analyze each cell and update the grid
for row in range(6):
    for col in range(7):
        x_start = col * cell_width
        y_start = row * cell_height
        grid[row, col] = process_cell(bilateral_filtered_image, x_start, y_start, cell_width, cell_height)

# Display the grid
print(grid)

def check_winner(grid):
    rows, cols = len(grid), len(grid[0])

    grid_array = np.array(grid)

    for row in grid_array:
        for i in range(cols - 3):
            if np.array_equal(row[i:i + 4], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(row[i:i + 4], np.array([-1, -1, -1, -1])):
                return -1

    for col in range(cols):
        for i in range(rows - 3):
            if np.array_equal(grid_array[i:i + 4, col], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(grid_array[i:i + 4, col], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(rows - 3):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(3, rows):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1
    return 0


result = check_winner(grid)
if result == 1:
    print("Player 1 (Red) wins!")
elif result == -1:
    print("Player 2 (Black) wins!")
else:
    print("The game is still ongoing or it's a draw.")

# grid to position string conversion

def grid_to_position_string(grid):
    turn = 1
    position_string = f''
    for col in range(7):
        for row in range(6):
            if grid[row][col] == -1:
                position_string += 'X'
                turn *= -1
            elif grid[row][col] == 1:
                position_string += 'O'
                turn *= -1
            else:
                position_string += '-'
    if turn > 0:
        turn = 'A'
    else:
        turn = 'B'
    position_string = f'R_{turn}_0_0_' + position_string
    return position_string


position_string = grid_to_position_string(grid)
print(position_string)


# only one move difference from grid to second_grid
def diff_one_move(grid, second_grid):
    first_grid_string = grid_to_position_string(grid)
    compare_string = first_grid_string[8:]
    second_grid_string = grid_to_position_string(second_grid)[8:]
    counter = 0
    for i in range(len(compare_string)):
        if compare_string[i] != second_grid_string[i]:
            break
        counter += 1
    counter = counter % 7
    return first_grid_string + f':{counter}'

