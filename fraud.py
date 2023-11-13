import cv2
import numpy as np


def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)


img = cv2.imread('deflated_warped_image.png')

bgr_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = bgr_image.reshape(-1, 3)

# Define color ranges
black_lower = np.array([0, 0, 0])
black_upper = np.array([80, 80, 80])

yellow_lower = np.array([150, 100, 0])
yellow_upper = np.array([255, 255, 80])

red_lower = np.array([100, 0, 0])
red_upper = np.array([255, 100, 100])

# Create masks
black_mask = cv2.inRange(bgr_image, black_lower, black_upper)
yellow_mask = cv2.inRange(bgr_image, yellow_lower, yellow_upper)
red_mask = cv2.inRange(bgr_image, red_lower, red_upper)

# Exclude dark blue
blue_lower = np.array([0, 0, 20])  # Adjust the lower bound to be darker
blue_upper = np.array([80, 80, 255])
blue_mask = cv2.inRange(bgr_image, blue_lower, blue_upper)


dominant_color_mask = black_mask + yellow_mask + red_mask - blue_mask


# Invert dominant color masks to get masks for other colors
other_color_mask = cv2.bitwise_not(dominant_color_mask)

white_mask = np.full_like(img, (255, 255, 255), dtype=np.uint8)
white_mask = cv2.bitwise_and(white_mask, white_mask, mask=other_color_mask)

result_image = cv2.bitwise_and(img, img, mask=dominant_color_mask)

result_image = cv2.bitwise_or(result_image, white_mask)


imgshow('Pranay', result_image)

cv2.imwrite('result_image2.png', result_image)


def divide_into_grid(image, rows, cols):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the size of each grid cell
    cell_height = height // rows
    cell_width = width // cols

    # Create a copy of the image to draw the grid
    grid_image = image.copy()

    # List to store the cell coordinates
    cell_coordinates = []

    # Iterate over each row and column to draw the grid and store the cell coordinates
    for row in range(rows):
        for col in range(cols):
            start_x = col * cell_width
            start_y = row * cell_height
            end_x = (col + 1) * cell_width
            end_y = (row + 1) * cell_height

            # Store the top-left and bottom-right coordinates of the cell
            cell_coordinates.append(((start_x, start_y), (end_x, end_y)))

            # Draw the cell on the grid image
            cv2.rectangle(grid_image, (start_x, start_y),
                          (end_x, end_y), (0, 255, 0), 1)

    return grid_image, cell_coordinates


# Load the image
image_path = 'result_image2.png'
grid_image = cv2.imread(image_path)

# Divide the image into a 6x7 grid and draw it
rows, cols = 6, 7
grid_image_with_cells, cells_coordinates = divide_into_grid(
    grid_image, rows, cols)

# Show the image with the grid
cv2.imshow('Grid Image', grid_image_with_cells)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with the grid
cv2.imwrite('grid_image_with_cells.png', grid_image_with_cells)
