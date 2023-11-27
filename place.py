import cv2
import numpy as np

def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)

image = cv2.imread('/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/ttt.jpeg')
imgshow("img", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

imgshow("thresh", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
xo_symbols = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 100 < area < 1000:  # Adjust the area threshold as needed
        xo_symbols.append(contour)

for symbol in xo_symbols:
    x, y, w, h = cv2.boundingRect(symbol)
    aspect_ratio = float(w) / h
    if 0.8 < aspect_ratio < 1.2:
        # Symbol is a circle (O)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        # Symbol is a cross (X)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Tic-Tac-Toe Board', image)
