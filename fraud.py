# Modules
import cv2
import numpy as np
# from sklearn.cluster import KMeans  # Import the KMeans class from scikit-learn

# Functions


def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)


img = cv2.imread('/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/deflated_warped_image.png')

bgr_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = bgr_image.reshape(-1, 3)

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(pixels)
# dominant_colors = kmeans.cluster_centers_

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


# # Convert to grayscale and threshold the image to create a binary mask
# gray_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
# _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# # Invert the binary mask to identify empty cells
# empty_mask = cv2.bitwise_not(binary_mask)

# # Replace the empty cells with pure white
# result_image = cv2.bitwise_and(result_image, result_image, mask=empty_mask)
# result_image[binary_mask == 255] = [255, 255, 255]


# Create a binary mask for the empty cells by inverting the color mask
# empty_mask = cv2.bitwise_not(dominant_color_mask)

# # Replace empty cells with pure white
# result_image2 = np.copy(img)
# result_image2[empty_mask > 0] = [255, 255, 255]



imgshow('Pranay', result_image)


cv2.imwrite('result_image2.png', result_image)


