import os
import cv2 as CV
import numpy as np

CWD = os.getcwd()

def read_image_file(fname):
    if os.path.isfile(f"{CWD}/images/{fname}"):
        img1 = CV.imread(f"{CWD}/images/{fname}")
        #img_code = CV.GraphicalCodeDetector.detect(img)
        #cyan = [66,138,179]
        img = game_board_color(BOARD_COLOR, img1)
        update_board(img1, PLAYER1_COLOR)
        #CV.moveWindow(fname, 50, 50)
        CV.namedWindow("Image", 0)
        CV.imshow("Image", img)
        CV.waitKey(0)
        
    else:
        pass

def find_color_image(color, img):
    lower_bound = tuple([x - 20 for x in color])
    upper_bound = tuple([x + 20 for x in color])
    masked = CV.inRange(img, lower_bound, upper_bound)
    img_masked = CV.bitwise_and(img, img, mask=masked)
    return img_masked

def is_coord_color(color, coord):
    lower_bound = tuple([x - 20 for x in color])
    upper_bound = tuple([x + 20 for x in color])
    print(lower_bound)
    masked = CV.inRange(coord, lower_bound, upper_bound)
    img_masked = CV.bitwise_and(coord, coord, mask=masked)
    points = CV.findNonZero(masked)
    

def game_board_color(board_color, img):
    lower_bound = tuple([x - 20 for x in board_color])
    upper_bound = tuple([x + 20 for x in board_color])
    #print(lower_bound)
    #print(upper_bound)
    #lower_bound = np.array([0, 100, 160])
    #upper_bound = np.array([90, 170, 220])
    # for i in range(0, len(img)):
    #     color = img[i]
    #     all_true = True
    #     for j in range(0, len(color)):
    #         for k in range(0, len(color[j])):
    #             if not(abs(color[j][k] - board_color[k]) <= 20) or (abs(color[j][k] + board_color[k]) <= 20):
    #                 all_true = False
    #                 break
    #         if all_true:
    #             img[i][j] = [255, 255, 255]
    #img_hsv = CV.cvtColor(img, CV.COLOR_BGR2RGBA)
    #img_hsv = CV.cvtColor(img, CV.viz.Color().cyan())
    #print(img_hsv)
    masked = CV.inRange(img, lower_bound, upper_bound)
    
    img_masked = CV.bitwise_and(img, img, mask=masked)
    points = CV.findNonZero(masked)
    game_board_dimensions(points, img)
    #CV.drawContours(img, contours, -1, (0, 255, 0), 15)
    return img

def game_board_dimensions(points, img_masked):
    #points = cv2.findNonZero(mask)
    #mode = np.median(points, axis=0)
    #print(mode)
    min_x = min([x[0][0] for x in points])
    min_y = min([x[0][1] for x in points])
    max_x = max([x[0][0] for x in points])
    max_y = max([x[0][1] for x in points])
    print(min_x, min_y, max_x, max_y)
    start = [min_x, min_y]
    end = [max_x, max_y]
    step_x = ((max_x - min_x) / GRID[0]) + 1
    step_y = ((max_y - min_y) / GRID[1]) + 1

    curr_x = start[0]
    curr_y = start[1]
    for i in range(0, len(BOARD)):
        coords = []
        for j in range(0, len(BOARD[i])):
            #print(int(curr_x))
            #print(int(curr_y + step_y))
            coords.append([curr_x + (step_x / 2), curr_y + (step_y / 2)])
            CV.line(img_masked, (int(curr_x), int(curr_y)), (int(curr_x), int(curr_y + step_y)), (0, 0, 255), 10)
            CV.line(img_masked, (int(curr_x), int(curr_y)), (int(curr_x + step_x), int(curr_y)), (0, 0, 255), 10)
            curr_y += step_y
        curr_y = start[1]
        curr_x += step_x
        PIXEL_BOARD.append(coords)
    CV.line(img_masked, (int(curr_x), int(curr_y)), (int(curr_x), int(end[1])), (0, 0, 255), 10)    
    CV.line(img_masked, (int(start[0]), int(end[1])), (int(curr_x), int(end[1])), (0, 0, 255), 10)    
    #CV.imshow(img_masked)

def update_board(img, player):
    for y in PIXEL_BOARD:
        for x in y:
            print(tuple(img[int(x[1])][int(x[0])]))
            is_coord_color(player, tuple(img[int(x[1])][int(x[0])]))
    pass

#game_board(cyan)
BOARD_COLOR = [168,122,45]
PLAYER1_COLOR = [236,220,59]
PLAYER2_COLOR = [239,154,67]
GRID_OFFSET = 50
GRID = (3, 3)
BOARD = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
PIXEL_BOARD = []
read_image_file("ttt.jpg")