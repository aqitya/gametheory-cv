from PIL import Image
import numpy as np

# Load the image from file
img_path = './images/frame_0.png' 
img = Image.open(img_path)
img = img.convert('RGB')

# Color Ranges
red_lower = (150, 0, 0)
red_upper = (255, 100, 100)
blue_lower = (0, 0, 150)
blue_upper = (100, 100, 255)

# Function to check if a pixel is within the color range
def is_color(pixel, lower, upper):
    return all(lower[i] <= pixel[i] <= upper[i] for i in range(3))

# 3x3 dodgem board
board_size = (3, 3) 
board = np.full(board_size, 0, dtype=int)

# Dividing the image into cells
width, height = img.size
cell_width = width // board_size[1]
cell_height = height // board_size[0]

for i in range(board_size[0]):
    for j in range(board_size[1]):
        center_x = j * cell_width + cell_width // 2
        center_y = i * cell_height + cell_height // 2
        pixel = img.getpixel((center_x, center_y))
        
        # Check the color of the center pixel and update the board representation
        if is_color(pixel, red_lower, red_upper):
            board[i, j] = -1
        elif is_color(pixel, blue_lower, blue_upper):
            board[i, j] = 1

# Convert the numpy array to a list for better readability
board_list = board.tolist()
print(board_list)

def board_to_string_with_turn(board_array):
    # Need to determine whose turn it is somehow...
    turn_char = 'A'
    
    string_repr = f"R_{turn_char}_4_4_----"
    
    for row in board_array:
        for cell in row:
            if cell == 1:
                string_repr += 'o'  # Blue
            elif cell == -1:
                string_repr += 'x'  # Red
            else:
                string_repr += '-'  # Empty space
        string_repr += '-'  # Add the dead character after each row
    
    return string_repr

# Test
new_board_array = [[0, 1, 0], [1, 0, 0], [0, -1, -1]]
string_representation_with_turn = board_to_string_with_turn(new_board_array)
print(string_representation_with_turn)