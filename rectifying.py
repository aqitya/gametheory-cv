# Modules
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(
    '/Users/aditummala/Desktop/gametheory-cv/images/IMG_0198.JPG')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Use Hough transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Draw the lines on the image
image_with_lines = img.copy()
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with lines drawn
plt.figure(figsize=(10, 10))
plt.imshow(image_with_lines)\
plt.axis('off')
plt.title('Image with Detected Lines')
plt.show()
