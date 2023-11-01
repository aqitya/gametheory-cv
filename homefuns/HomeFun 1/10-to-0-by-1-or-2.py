# Implementing the game-specific functions

def DoMove(position, move):
    return position - move


def GenerateMoves(position):
    if position >= 2:
        return {1, 2}
    elif position == 1:
        return {1}
    else:
        return {}


def PrimitiveValue(position):
    if position == 0:
        return "lose"
    else:
        return "not_primitive"
