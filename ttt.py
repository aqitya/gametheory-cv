import cv2
import numpy as np

def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)



image = cv2.imread('/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/ttt.jpeg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
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
    
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the lines on the original image

# # cv2.imwrite('output_image.png', image)  # Save the image with the boxes drawn

imgshow("ttt", image)

#create a 2d array to hold the gamestate
gamestate = [["-","-","-"],["-","-","-"],["-","-","-"]]

#kernel used for noise removal
kernel =  np.ones((7,7),np.uint8)
# Load a color image 
img =image
# get the image width and height
img_width = img.shape[0]
img_height = img.shape[1]

# turn into grayscale
img_g =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# turn into thresholded binary
ret,thresh1 = cv2.threshold(img_g,127,255,cv2.THRESH_BINARY)
#remove noise from binary
thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

#find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
im2, contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if the version of OpenCV returns image as the first result (newer versions)
if len(contours) == 3:
    _, contours, _ = contours  # Extract the contours from the result

cv2.drawContours(img, contours, -1, (0, 255, 0), 15)

tileCount = 0
for cnt in contours:
        # ignore small contours that are not tiles
        if cv2.contourArea(cnt) > 200000: 
                tileCount = tileCount+1
                # use boundingrect to get coordinates of tile
                x,y,w,h = cv2.boundingRect(cnt)
                # create new image from binary, for further analysis. Trim off the edge that has a line
                tile = thresh1[x+40:x+w-80,y+40:y+h-80]
                # create new image from main image, so we can draw the contours easily
                imgTile = img[x+40:x+w-80,y+40:y+h-80]

                #determine the array indexes of the tile
                tileX = round((x/img_width)*3)
                tileY = round((y/img_height)*3)     

                # find contours in the tile image. RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours.
                im2, c, hierarchy = cv2.findContours(tile, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
                for ct in c:
                        # to prevent the tile finding itself as contour
                        if cv2.contourArea(ct) < 180000:
                                cv2.drawContours(imgTile, [ct], -1, (255,0,0), 15)
                                #calculate the solitity
                                area = cv2.contourArea(ct)
                                hull = cv2.convexHull(ct)
                                hull_area = cv2.contourArea(hull)
                                solidity = float(area)/hull_area

                                # fill the gamestate with the right sign
                                if(solidity > 0.5):
                                        gamestate[tileX][tileY] = "O"
                                else: 
                                        gamestate[tileX][tileY] = "X"
                # put a number in the tile
                cv2.putText(img, str(tileCount), (x+200,y+300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)

#print the gamestate
print("Gamestate:")
for line in gamestate:
        linetxt = ""
        for cel in line:
                linetxt = linetxt + "|" + cel
        print(linetxt)

# resize final image
res = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)

# display image and release resources when key is pressed
cv2.imshow('image1',res)
cv2.waitKey(0)
cv2.destroyAllWindows()