# Implementing the game-specific functions

def DoMove(position, move):
    return position - move


def GenerateMoves(position):
    if position >= 5:
        return {1, 3, 4}
    elif position == 1:
        return {1}
    elif position == 3:
        return {3}
    elif position == 4:
        return {4}
    else:
        return {}


def PrimitiveValue(position):
    if position == 0:
        return "lose"
    else:
        return "not_primitive"
