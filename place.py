import cv2
import numpy as np

image = cv2.imread(
    '/Users/aditummala/Desktop/gametheory-cv/images/connect4.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edges = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the board
contour = max(contours, key=cv2.contourArea)


# Assuming board is a rectangle and we have its four corner points
# For simplicity, let's assume the contour approximated to a rectangle.
rect = cv2.boundingRect(contour)
x, y, w, h = rect

cell_width = w / 7
cell_height = h / 6

grid = []

for i in range(6):
    row = []
    for j in range(7):
        cell_x = int(x + j * cell_width)
        cell_y = int(y + i * cell_height)

        # Extract the cell from the original image
        cell = image[cell_y:cell_y +
                     int(cell_height), cell_x:cell_x+int(cell_width)]

        # Check the color in the center of the cell
        center_color = cell[int(cell_height/2), int(cell_width/2)]

        # Determine the piece color based on the BGR values
        if np.allclose(center_color, [0, 255, 255], atol=50):  # black
            row.append(1)
        elif np.allclose(center_color, [0, 0, 255], atol=50):  # red
            row.append(-1)
        else:
            row.append(0)
    grid.append(row)

print(grid)
