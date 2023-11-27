import cv2
import numpy as np

# Function to read an image and convert it to HSV
def read_and_convert_to_hsv(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

# Function to create a mask for a given color range
def create_mask_for_color(hsv, lower_color, upper_color):
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

# Function to determine the position of the mark within the cell
def position_in_cell(hsv, mask, cell_positions):
    for position in cell_positions:
        x_start, y_start, x_end, y_end = position
        cell_hsv = hsv[y_start:y_end, x_start:x_end]
        cell_mask = mask[y_start:y_end, x_start:x_end]
        
        # If the majority of the cell is the color, then we say the mark is in that cell
        if np.sum(cell_mask) > (cell_mask.size // 2):
            cv2.rectangle(hsv, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
            return True
    return False

# Function to print the game state
def print_game_state(game_state):
    for row in game_state:
        print('|'.join(row))

# Process the image and identify the game state
def identify_game_state(image_path):
    hsv_img = read_and_convert_to_hsv(image_path)
    
    # Define color ranges for blue and red
    lower_blue = np.array([90, 50, 50])  
    upper_blue = np.array([120, 255, 255]) 
    # lower_red = np.array([0, 30, 30])  
    # upper_red = np.array([40, 255, 255])  

    # Create masks for blue and red
    mask_blue = create_mask_for_color(hsv_img, lower_blue, upper_blue)
    # mask_red = create_mask_for_color(hsv_img, lower_red, upper_red)

    # Calculate cell positions assuming a 3x3 grid
    cell_size_x = hsv_img.shape[1] // 3
    cell_size_y = hsv_img.shape[0] // 3
    cell_positions = [(x * cell_size_x, y * cell_size_y, (x+1) * cell_size_x, (y+1) * cell_size_y) for y in range(3) for x in range(3)]

    # Initialize the game state
    game_state = [["-"] * 3 for _ in range(3)]

    # Determine the position of 'X' and 'O' within the cells
    for i in range(3):
        for j in range(3):
            position = cell_positions[i*3+j]
            if position_in_cell(hsv_img, mask_blue, [position]):
                game_state[i][j] = 'X'
            # elif position_in_cell(hsv_img, mask_red, [position]):
            #     game_state[i][j] = 'O'

    return game_state

# Run the game state identification
image_path = '/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/tica3.jpeg'
game_state_x = identify_game_state(image_path)
print("Gamestate x:")
print_game_state(game_state_x)


##########################################################################
#GAME BOARD FOR O
###########################################################################
# Read the image
img = cv2.imread('/Users/pranayrajpaul/Desktop/gamescrafters/gametheory-cv/images/tica3.jpeg')

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define HSV range for light blue (X)
# lower_blue = np.array([100, 150, 150])
# upper_blue = np.array([130, 255, 255])

# Define HSV range for dark red (O)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Create masks for blue and red
# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Function to find contours and filter by area size
def find_significant_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small contours that are likely not 'X' or 'O'
    return [cnt for cnt in contours if cv2.contourArea(cnt) > img.shape[0] * img.shape[1] * 0.01]

# Get contours for X and O
# contours_blue = find_significant_contours(img, mask_blue)
contours_red = find_significant_contours(img, mask_red)

# Initialize game state
gamestate_o = [["-","-","-"],["-","-","-"],["-","-","-"]]

# Assume a 3x3 grid for the tic-tac-toe board
cell_size = img.shape[0] // 3

# Function to determine the grid index based on the contour centroid
def get_grid_index(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0: return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cy // cell_size, cx // cell_size)

# # Fill in the gamestate with 'X'
# for cnt in contours_blue:
#     index = get_grid_index(cnt)
#     if index:
#         gamestate_o[index[0]][index[1]] = "X"

# Fill in the gamestate with 'O'
for cnt in contours_red:
    index = get_grid_index(cnt)
    if index:
        gamestate_o[index[0]][index[1]] = "O"

# Print the gamestate
print("Gamestate o:")
print_game_state(gamestate_o)


##########################################################################
#Post Processing
###########################################################################


combined_gamestate = [["-"] * 3 for _ in range(3)]

# Iterate over the rows and columns to combine the game states
for i in range(3):
    for j in range(3):
        if game_state_x[i][j] != "-":  # If there's an 'X' in the cell, take it
            combined_gamestate[i][j] = game_state_x[i][j]
        elif gamestate_o[i][j] != "-":  # If there's an 'O' in the cell, take it
            combined_gamestate[i][j] = gamestate_o[i][j]
        # If both are '-', it stays '-'

# Print the combined game state
print("Combined Gamestate:")
print_game_state(combined_gamestate)

def convert_gamestate_to_pos_str(gamestate):
    countx = 0
    counto = 0
    turn = 'A'
    for i in range(3):
        for j in range(3):
            if gamestate[i][j] == "X":  
                countx +=1
            if gamestate[i][j] == "O":  
                counto +=1
    print(countx)
    print(counto)

    if countx > counto:
        turn = 'B'
    else:
        turn = 'A'
    position_str = "R_"+turn+"_3_3_"
    for i in range(3):
        for j in range(3):
            if gamestate[i][j] == "X":  
                position_str += 'x'
            if gamestate[i][j] == "O":  
                position_str += 'o'
            if gamestate[i][j] == "-":
                position_str += '-'
    return position_str
 

 


pos_string = convert_gamestate_to_pos_str(combined_gamestate)
print("Position String:")
print(pos_string)

