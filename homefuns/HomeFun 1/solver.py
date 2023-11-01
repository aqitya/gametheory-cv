import TicTacToe as game

cache = {}


def solve(position):
    pos_tuple = tuple(tuple(row) for row in position)

    if pos_tuple in cache:
        return cache[pos_tuple]

    primVal = game.PrimitiveValue(position)
    if primVal != 'not_primitive':
        cache[pos_tuple] = primVal
        return primVal

    moves = game.GenerateMoves(position)
    outcomes = []
    for move in moves:
        next_position = game.doMove(position, move)
        outcome = solve(next_position)
        outcomes.append(outcome)

    if 'lose' in outcomes:
        result = 'win'
    elif 'tie' in outcomes:
        result = 'tie'
    else:
        result = 'lose'

    cache[pos_tuple] = result
    return result


board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

solve(board)


def count():
    win_count = 0
    lose_count = 0
    tie_count = 0
    primitive_lose = 0
    primitive_tie = 0
    primitive_win = 0

    for board, outcome in cache.items():
        board = [list(row) for row in board]
        prim_outcome = game.PrimitiveValue(board)
        if outcome == 'win':
            win_count += 1
        elif outcome == 'lose':
            lose_count += 1
        elif outcome == 'tie':
            tie_count += 1
        if prim_outcome == 'lose':
            primitive_lose += 1
        elif prim_outcome == 'tie':
            primitive_tie += 1
        elif prim_outcome == 'win':
            primitive_win += 1
    print(f'Lose: {lose_count} ({primitive_lose} prmitive)')
    print(f'Win: {win_count} ({primitive_win} primitive)')
    print(f'Tie: {tie_count} ({primitive_tie} primitive)')
    print(f'Total: {lose_count + win_count + tie_count} ({primitive_lose + primitive_tie + primitive_win} primitiv)')


count()
