# Modules
import cv2
import numpy as np

# Functions


def imgshow(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 200, 200)
    cv2.waitKey(0)


img = cv2.imread('/Users/aditummala/Desktop/gametheory/homefuns/connect4.jpeg')

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
    approx = cv2.approxPolyDP(
        contour, 0.01*cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)

    rect = cv2.boundingRect(contour)
    x_rect, y_rect, w_rect, h_rect = rect
    x_rect += w_rect/2
    y_rect += h_rect/2
    area_rect = w_rect*h_rect

    if ((len(approx) > 8) & (len(approx) < 23) & (area > 250) & (area_rect < (img_w*img_h)/5)) & (w_rect in range(h_rect-10, h_rect+10)):  # Circle conditions
        contour_list.append(contour)
        position_list.append((x_rect, y_rect))
        rect_list.append(rect)

img_circle_contours = img_orig.copy()
cv2.drawContours(img_circle_contours, contour_list,  -1,
                 (0, 255, 0), thickness=1)  # Display Circles
for rect in rect_list:
    x, y, w, h = rect
    cv2.rectangle(img_circle_contours, (x, y), (x+w, y+h), (0, 0, 255), 1)

imgshow('Circles Detected', img_circle_contours)

# Interpolate Grid
rows, cols = (6, 7)
mean_w = sum([rect[2] for r in rect_list]) / len(rect_list)
mean_h = sum([rect[3] for r in rect_list]) / len(rect_list)
position_list.sort(key=lambda x: x[0])
max_x = int(position_list[-1][0])
min_x = int(position_list[0][0])
position_list.sort(key=lambda x: x[1])
max_y = int(position_list[-1][1])
min_y = int(position_list[0][1])
grid_width = max_x - min_x
grid_height = max_y - min_y
col_spacing = int(grid_width / (cols-1))
row_spacing = int(grid_height / (rows - 1))

# Find Masks
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([150, 150, 100])
upper_red = np.array([255, 255, 255])
mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
img_red = cv2.bitwise_and(img, img, mask=mask_red)
imgshow("Red Mask", img_red)

lower_yellow = np.array([10, 150, 100])
upper_yellow = np.array([60, 255, 255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
img_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
imgshow("Yellow Mask", img_yellow)

# Identify Colours
grid = np.zeros((rows, cols))
id_red = 1
id_yellow = -1
img_grid_overlay = img_orig.copy()
img_grid = np.zeros([img_h, img_w, 3], dtype=np.uint8)

for x_i in range(0, cols):
    x = int(min_x + x_i * col_spacing)
    for y_i in range(0, rows):
        y = int(min_y + y_i * row_spacing)
        r = int((mean_h + mean_w)/5)
        img_grid_circle = np.zeros((img_h, img_w))
        cv2.circle(img_grid_circle, (x, y), r, (255, 255, 255), thickness=-1)
        img_res_red = cv2.bitwise_and(
            img_grid_circle, img_grid_circle, mask=mask_red)
        img_grid_circle = np.zeros((img_h, img_w))
        cv2.circle(img_grid_circle, (x, y), r, (255, 255, 255), thickness=-1)
        img_res_yellow = cv2.bitwise_and(
            img_grid_circle, img_grid_circle, mask=mask_yellow)
        cv2.circle(img_grid_overlay, (x, y), r, (0, 255, 0), thickness=1)
        if img_res_red.any() != 0:
            grid[y_i][x_i] = id_red
            cv2.circle(img_grid, (x, y), r, (0, 0, 255), thickness=-1)
        elif img_res_yellow.any() != 0:
            grid[y_i][x_i] = id_yellow
            cv2.circle(img_grid, (x, y), r, (0, 255, 255), thickness=-1)

print('Grid Detected:\n', grid)
imgshow('Img Grid Overlay', img_grid_overlay)
imgshow('Img Grid', img_grid)

# Find Winner


def check_winner(grid):
    rows, cols = len(grid), len(grid[0])

    grid_array = np.array(grid)

    for row in grid_array:
        for i in range(cols - 3):
            if np.array_equal(row[i:i + 4], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(row[i:i + 4], np.array([-1, -1, -1, -1])):
                return -1

    for col in range(cols):
        for i in range(rows - 3):
            if np.array_equal(grid_array[i:i + 4, col], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(grid_array[i:i + 4, col], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(rows - 3):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(3, rows):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1
    return 0


result = check_winner(grid)
if result == 1:
    print("Player 2 (Red) wins!")
elif result == -1:
    print("Player 1 (Yellow) wins!")
else:
    print("The game is still ongoing or it's a draw.")

# grid to position string conversion


def grid_to_position_string(grid):
    turn = 1
    position_string = f''
    for col in range(cols):
        for row in range(rows):
            if grid[row][col] == -1:
                position_string += 'X'
                turn *= -1
            elif grid[row][col] == 1:
                position_string += 'O'
                turn *= -1
            else:
                position_string += '-'
    if turn > 0:
        turn = 'A'
    else:
        turn = 'B'
    position_string = f'R_{turn}_0_0_' + position_string
    return position_string


position_string = grid_to_position_string(grid)
print(position_string)

# only one move difference from grid to second_grid


def diff_one_move(grid, second_grid):
    first_grid_string = grid_to_position_string(grid)
    compare_string = first_grid_string[8:]
    second_grid_string = grid_to_position_string(second_grid)[8:]
    counter = 0
    for i in range(len(compare_string)):
        if compare_string[i] != second_grid_string[i]:
            break
        counter += 1
    counter = counter % 7
    return first_grid_string + f':{counter}'


grid = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0], [0, 1, 1, 1, -1, 0, 0],
        [0, -1, -1, -1, 1, 0, 0], [0, -1, 1, 1, 1, 0, 0], [1, -1, -1, -1, 1, -1, 0]]

second_grid = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0], [0, 1, 1, 1, -1, 0, 0],
               [0, -1, -1, -1, 1, 0, 0], [0, -1, 1, 1, 1, 1, 0], [1, -1, -1, -1, 1, -1, 0]]

print(diff_one_move(grid, second_grid))
