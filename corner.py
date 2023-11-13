import cv2
import numpy as np
from scipy.spatial import distance as dist

# Adjusting the green color range based on typical green hue values in HSV
# Hue values for green usually fall between 40 and 80 in OpenCV
tighter_green_range = np.array([[40, 0, 127], [80, 243, 255]])


def find_specific_green_objects(image, color_range):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only colors in the specified range
    mask = cv2.inRange(hsv, color_range[0], color_range[1])

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the centroid of each contour and draw it on the image
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            # Draw the contour and centroid on the image
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

    return image, centroids


# Load the main image again
main_image_specific_green = cv2.imread('./images/new_emeralds.png')
if main_image_specific_green is None:
    raise FileNotFoundError("The main image could not be loaded.")

# Find specific green objects and their centroids
result_image_specific_green, specific_green_centroids = find_specific_green_objects(
    main_image_specific_green, tighter_green_range)

# Save the result
result_image_specific_green_path = 'result_with_specific_green_centroids.png'
cv2.imwrite(result_image_specific_green_path, result_image_specific_green)

specific_green_centroids, result_image_specific_green_path

# We will modify the function to find centroids to only consider the four largest green areas by area.


def find_largest_green_centroids(image, color_range, num_largest=4):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only colors in the specified range
    mask = cv2.inRange(hsv, color_range[0], color_range[1])

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area and grab the largest ones
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[
        :num_largest]

    # Calculate the centroid of each of the largest contours and draw it on the image
    largest_centroids = []
    for cnt in largest_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            largest_centroids.append((cX, cY))
            # Draw the contour and centroid on the image
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

    return image, largest_centroids


# Use the function to find the centroids of the four largest green areas
result_image_largest_green, largest_green_centroids = find_largest_green_centroids(
    main_image_specific_green.copy(), tighter_green_range)

# Save the result
result_image_largest_green_path = 'result_with_specific_green_centroids.png'
cv2.imwrite(result_image_largest_green_path, result_image_largest_green)

print(largest_green_centroids)


src_pts = np.array(largest_green_centroids, dtype='float32')

# Find the minimum and maximum x and y coordinates from the centroids
min_x = min(c[0] for c in largest_green_centroids)
max_x = max(c[0] for c in largest_green_centroids)
min_y = min(c[1] for c in largest_green_centroids)
max_y = max(c[1] for c in largest_green_centroids)

# Use these coordinates to define the bounding rectangle and extract the ROI
roi = main_image_specific_green[min_y:max_y, min_x:max_x]

# Save the extracted ROI
roi_image_path = '/mnt/data/extracted_roi.png'
cv2.imwrite(roi_image_path, roi)
width = max_x - min_x
height = max_y - min_y

# Redefine the order_points function with scipy.spatial.distance


def order_points(pts):
    # Sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Grab the left-most and right-most points from the sorted x-coordinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Now, sort the left-most coordinates according to their y-coordinates
    # to grab the top-left and bottom-left points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # Now that we have the top-left point, with a simple calculation we can find the bottom-right and top-right points
    # Calculate the Euclidean distance between the top-left and right-most points
    # The point with the maximum distance will be the bottom-right point
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    # Return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


# Use the order_points function to arrange the centroids properly
ordered_pts = order_points(np.array(largest_green_centroids))

# Now we can perform the perspective warp with the ordered points
# The destination points will be a rectangle based on the maximum width and height we found
dst_pts = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")


def deflate_points(points, deflate_amount):
    # Calculate the center of the rectangle
    center_x = np.mean([point[0] for point in points])
    center_y = np.mean([point[1] for point in points])

    # Move each point towards the center by the deflate amount
    new_points = []
    for point in points:
        if point[0] < center_x:
            new_x = point[0] + deflate_amount
        else:
            new_x = point[0] - deflate_amount

        if point[1] < center_y:
            new_y = point[1] + deflate_amount
        else:
            new_y = point[1] - deflate_amount

        new_points.append((new_x, new_y))

    return np.array(new_points, dtype='float32')


# Deflate the points by 50 pixels
deflated_pts = deflate_points(ordered_pts, 25)

# The destination points will remain the same as before (a rectangle)
# Compute the new perspective transform matrix with the deflated points
M_deflated = cv2.getPerspectiveTransform(deflated_pts, dst_pts)

# Warp the perspective to transform the image using the deflated points
deflated_warped_image = cv2.warpPerspective(
    main_image_specific_green, M_deflated, (width, height))

# Save the deflated warped image
deflated_warped_image_path = 'deflated_warped_image.png'
cv2.imwrite(deflated_warped_image_path, deflated_warped_image)

deflated_warped_image_path
