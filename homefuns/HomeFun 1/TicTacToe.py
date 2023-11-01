def doMove(board, move):
    position, player = move
    x, y = position
    blank = []
    for i in board:
        new_row = []
        for j in i:
            new_row.append(j)
        blank.append(new_row)
    blank[x][y] = player
    board = blank
    return board


def GenerateMoves(board):
    player_counts = []
    for row in board:
        player_counts.append(sum(row))

    total = 0
    for count in player_counts:
        total += count

    if total <= 0:
        player = 1
    else:
        player = -1

    moves = []
    for x in range(3):
        for y in range(3):
            if board[x][y] == 0:
                move = ((x, y), player)
                moves.append(move)

    return moves


def PrimitiveValue(board):
    counts = []
    for row in board:
        counts.append(sum(row))

    total = sum(counts)
    if total <= 0:
        next_player = 1
    else:
        next_player = -1
    other_player = -next_player

    for row in board:
        row_win = True
        for cell in row:
            if cell != other_player:
                row_win = False
                break
        if row_win:
            return 'lose'

    for j in range(3):
        col_win = True
        for i in range(3):
            if board[i][j] != other_player:
                col_win = False
                break
        if col_win:
            return 'lose'

    diag1 = True
    for i in range(3):
        if board[i][i] != other_player:
            diag1 = False
            break
    if diag1:
        return 'lose'

    diag2 = True
    for i in range(3):
        if board[i][2 - i] != other_player:
            diag2 = False
            break
    if diag2:
        return 'lose'

    tie = True
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                tie = False
                break
        if not tie:
            break

    if tie:
        return 'tie'

    return 'not_primitive'
