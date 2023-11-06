# Modules
import cv2
import numpy as np

# Functions


def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)


img = cv2.imread(
    '/Users/aditummala/Desktop/gametheory-cv/images/centered.jpg')

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

# Outline Edges
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 150)
imgshow('Edge Detection', edge_detected_image)

# Find Circles
contours, hierarchy = cv2.findContours(
    edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_list = []
rect_list = []
position_list = []

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:  # To avoid division by zero
        continue
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    # Define a minimum circularity threshold for what you consider a circle
    # This can be adjusted based on how lenient you want to be
    circularity_threshold = 0.4

    # Optionally, you could also check the size of the area
    min_area_threshold = 50  # This can be adjusted based on your specific needs

    if circularity > circularity_threshold and area > min_area_threshold:
        contour_list.append(contour)
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            position_list.append((cX, cY))

print(position_list)


def find_connect_four_grid(contour_list, expected_rows=6, expected_columns=7, tolerance=0.2):
    """
    Try to find a grid pattern in the detected circles.

    :param contour_list: List of contours that are potential circles.
    :param expected_rows: Expected number of rows in the grid.
    :param expected_columns: Expected number of columns in the grid.
    :param tolerance: Allowed deviation from the mean distance to consider as a grid.
    :return: Grid of circles if found, otherwise None.
    """
    # Calculate the centroids of the contours
    centroids = []
    for contour in contour_list:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    # Sort the centroids, first by y-coordinate, then by x-coordinate
    centroids = sorted(centroids, key=lambda x: (x[1], x[0]))

    # Try to align centroids to a grid
    # Starting with the first centroid, check if there are expected_columns - 1 more centroids
    # within a certain x-distance, and expected_rows - 1 more centroids within a certain y-distance
    for i, centroid in enumerate(centroids):
        grid = [centroid]
        x0, y0 = centroid

        # Find neighbors in the same row
        for dx in range(1, expected_columns):
            neighbor = next((c for c in centroids if abs(
                c[0] - (x0 + dx * mean_x_dist)) < tolerance * mean_x_dist and abs(c[1] - y0) < tolerance * mean_y_dist), None)
            if neighbor:
                grid.append(neighbor)

        # If we found a full row, try to build the full grid
        if len(grid) == expected_columns:
            full_grid = [grid]
            for dy in range(1, expected_rows):
                row = []
                for dx in range(expected_columns):
                    neighbor = next((c for c in centroids if abs(c[0] - (grid[dx][0])) < tolerance * mean_x_dist and abs(
                        c[1] - (grid[0][1] + dy * mean_y_dist)) < tolerance * mean_y_dist), None)
                    if neighbor:
                        row.append(neighbor)
                if len(row) == expected_columns:
                    full_grid.append(row)
                else:
                    break  # Row not complete, grid not valid

            # If we built a full grid, return it
            if len(full_grid) == expected_rows:
                return full_grid

    # If we reach this point, we didn't find a valid grid
    return None


# Calculate mean distances for x and y between circles
x_distances = []
y_distances = []
for i, (x1, y1) in enumerate(position_list[:-1]):
    for x2, y2 in position_list[i+1:]:
        x_distances.append(abs(x1 - x2))
        y_distances.append(abs(y1 - y2))

# Finding a grid from nothing does not involve anything else at all.


# Instead, do the following to ensure that there is nothing.

mean_x_dist = np.mean([dist for dist in x_distances if dist != 0])
mean_y_dist = np.mean([dist for dist in y_distances if dist != 0])

# Now we can try to find the grid
grid = find_connect_four_grid(contour_list)

if grid and len(grid) == 42:
    print("Found a Connect 4 grid with the correct number of circles.")
else:
    print("Did not find a valid Connect 4 grid.")


img_circle_contours = img_orig.copy()
cv2.drawContours(img_circle_contours, contour_list,  -1,
                 (0, 255, 0), thickness=1)  # Display Circles
for rect in rect_list:
    x, y, w, h = rect
    cv2.rectangle(img_circle_contours, (x, y), (x+w, y+h), (0, 0, 255), 1)

imgshow('Circles Detected', img_circle_contours)
