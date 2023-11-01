def DoMove(position, move):
    new_position = [row.copy() for row in position]
    new_position[move[0][0]][move[0][1]] = move[1]
    return new_position


def PrimitiveValue(position):
    for i in range(len(position)):
        for j in range(len(position[i])):
            if position[i][j] == 0:
                return "not_primitive"
    count = (0, 0)
    for i in range(3):
        for j in range(3):
            if position[i][j] == -1:
                count[0] += 1
            elif position[i][j] == 1:
                count[1] += 1
    symbol = 0
    if count[0] > count[1]:
        symbol = -1
    elif count[0] < count[1]:
        symbol = 1
    for i in range(3):
        if position[i][0] == position[i][1] == position[i][2] == symbol:
            return "lose"
        if position[0][i] == position[1][i] == position[2][i] == symbol:
            return "lose"
    if position[0][0] == position[1][1] == position[2][2] == symbol:
        return "lose"
    if position[0][2] == position[1][1] == position[2][0] == symbol:
        return "lose"
    return "tie"
