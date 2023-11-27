import numpy as np
import cv2

def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)


#create a 2d array to hold the gamestate
gamestate = [["-","-","-"],["-","-","-"],["-","-","-"]]

#kernel used for noise removal
kernel =  np.ones((7,7),np.uint8)
# Load a color image 
img = cv2.imread('/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/tica.jpeg')
# get the image width and height
img_width = img.shape[0]
img_height = img.shape[1]

# turn into grayscale
img_g =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# turn into thresholded binary
ret, thresh1 = cv2.threshold(img_g, 127, 255, cv2.THRESH_BINARY)
#remove noise from binary
thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

#find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours

imgshow('Grayscale Image', img_g)
imgshow('Thresholded Image', thresh1)


contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    cv2.drawContours(img, contours, -1, (0, 255, 0), 15)
else:
    print("No contours found in the image.")


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
imgshow('image1',res)
