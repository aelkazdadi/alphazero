from typing import List
from chess_constants import INIT_STATE, symbols
from chess_constants import EMPTY, WHITE_KING,\
    WHITE_BISHOP,\
    WHITE_QUEEN, WHITE_KNIGHT, WHITE_ROOK,\
    WHITE_PAWN, WHITE_PAWN_EN_PASSANT,\
    BLACK_PAWN, BLACK_PAWN_EN_PASSANT,\
    WHITE_MIN, WHITE_MAX,\
    BLACK_MIN, BLACK_MAX


class State:
    def __init__(self, state_vector: List):
        self.state: List[int]
        self.state = state_vector

    def __str__(self) -> str:
        string = ""
        for row in range(7, -1, -1):
            for column in range(8):
                unit = self.state[column + 8*row]
                string = string + " " + symbols[unit]
            if row > 0:
                string = string + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "State") -> bool:
        return self.state == other.state

    def do(self, action: "Action") -> "State":
        """Apply action to current state
        Returns next state, from the perspective of the other player.
        """
        from_square = action.from_square
        to_square = action.to_square

        new_state = self.state.copy()

        # Disallow castling
        if self.state[from_square] == WHITE_KING:
            new_state[-1] = False
            new_state[-2] = False

        elif self.state[from_square] == WHITE_ROOK:
            if from_square == 0:
                new_state[-1] = False
            elif to_square == 7:
                new_state[-2] = False

        if self.state[from_square] == WHITE_KING and\
                self.state[to_square] == WHITE_ROOK:
            # Castling
            if to_square == 7:
                new_state[from_square] = EMPTY
                new_state[to_square] = EMPTY
                new_state[6] = WHITE_KING
                new_state[5] = WHITE_ROOK
            elif to_square == 0:
                new_state[from_square] = EMPTY
                new_state[to_square] = EMPTY
                new_state[2] = WHITE_KING
                new_state[3] = WHITE_ROOK

        elif (self.state[from_square] == WHITE_PAWN and
                8 <= from_square < 16 and
                to_square == from_square + 16):
            # En passant
            new_state[from_square] = EMPTY
            new_state[to_square] = WHITE_PAWN_EN_PASSANT

        elif (self.state[to_square] == EMPTY and
                self.state[from_square] == WHITE_PAWN and
                self.state[to_square - 8] == BLACK_PAWN_EN_PASSANT):
            # En passant capture
            new_state[from_square] = EMPTY
            new_state[to_square] = WHITE_PAWN

        else:
            # Standard move
            new_state[to_square] = self.state[from_square]
            new_state[from_square] = EMPTY

        # Disallow en passant (Black pawns)
        for i in range(8):
            if new_state[32+i] == BLACK_PAWN_EN_PASSANT:
                new_state[32+i] = BLACK_PAWN

        # Promoting pawns
        if new_state[to_square] == WHITE_PAWN and\
                56 <= to_square < 64:
            new_state[to_square] = action.promotion

        return State(new_state)

    def get_actions(self) -> List["Action"]:
        """Returns all the legal actions from the current state,
        from the perspective of White."""
        actions = []
        for row in range(8):
            for col in range(8):
                pos = 8*row + col
                unit = self.state[pos]

                if WHITE_MIN <= unit < WHITE_MAX:
                    # Pawn moves
                    if unit == WHITE_PAWN:
                        if self.state[pos+8] == EMPTY:
                            if 48 <= pos < 56:
                                # Promotion move
                                actions.append(Action(pos, pos+8, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+8, WHITE_KNIGHT))

                            else:
                                if 8 <= pos < 16:
                                    # First move
                                    actions.append(
                                        Action(pos, pos+16,
                                               WHITE_PAWN_EN_PASSANT))
                                # Standard move
                                actions.append(Action(pos, pos+8, EMPTY))

                        if (BLACK_MIN <= self.state[pos+9] < BLACK_MAX or
                            self.state[pos+1] == BLACK_PAWN_EN_PASSANT) and\
                                (pos % 8) != 7:
                            # Capture
                            if 48 <= pos < 56:
                                # Capture and promotion
                                actions.append(Action(pos, pos+9, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+9, WHITE_KNIGHT))
                            else:
                                # Standard capture
                                actions.append(Action(pos, pos+9, EMPTY))

                        if (BLACK_MIN <= self.state[pos+9] < BLACK_MAX or
                            self.state[pos-1] == BLACK_PAWN_EN_PASSANT) and\
                                (pos % 8) != 0:
                            if 48 <= pos < 56:
                                # Capture and promotion
                                actions.append(Action(pos, pos+7, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+7, WHITE_KNIGHT))
                            else:
                                # Standard capture
                                actions.append(Action(pos, pos+7, EMPTY))
                    elif unit == WHITE_ROOK:
                        # Rook moves
                        for k in (-1, 1, 8, 8):
                            i = 0
                            while True:
                                i += 1
                                if k == -1 or k == 1:
                                    in_bounds = (8*row <= pos + k*i
                                                 < 8*(row+1))
                                else:
                                    in_bounds = (0 <= pos + k*i < 64)

                                if in_bounds:
                                    if self.state[pos + k*i] == EMPTY:
                                        actions.append(
                                            Action(pos, pos + k*i, EMPTY))
                                        continue
                                    elif BLACK_MIN <= self.state[pos + k*i]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + k*i, EMPTY))
                                        break
                                    break
                                else:
                                    break
                    elif unit == WHITE_BISHOP:
                        # Bishop moves
                        for k in ((-1, -8), (-1, 8), (1, -8), (1, 8)):
                            i = 0
                            while True:
                                i += 1
                                in_bounds = (0 <= pos + k[1]*i < 64) and\
                                    (8*row <= pos + k[0]*i < 8*(row+1))

                                if in_bounds:
                                    step = k[0] + k[1]
                                    if self.state[pos + step*i] == EMPTY:
                                        actions.append(
                                            Action(pos, pos + step*i, EMPTY))
                                        continue
                                    elif BLACK_MIN <= self.state[pos + step*i]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + step*i, EMPTY))
                                        break
                                    break
                                else:
                                    break
                    elif unit == WHITE_KNIGHT:
                        # Knight moves
                        for (i, j) in ((1, 2), (2, 1), (-1, 2), (-2, 1),
                                       (-2, -1), (-1, -2), (2, -1), (1, -2)):
                            in_bounds = (0 <= row + j < 8) and\
                                    (0 <= col + i < 8)
                            target = self.state[pos + i + 8*j]
                            if in_bounds and (target == EMPTY or
                                              BLACK_MIN <= target < BLACK_MAX):
                                actions.append(
                                    Action(pos, pos + i + 8*j, EMPTY))

                    elif unit == WHITE_QUEEN:
                        # Queen moves
                        for (i, j) in ((1, 0), (1, 1), (0, 1), (-1, 1),
                                       (-1, 0), (-1, -1), (0, -1), (1, -1)):
                            k = 0
                            while True:
                                k += 1
                                in_bounds = ((0 <= row + k*j < 8) and
                                             (0 <= col + k*i < 8))
                                if in_bounds:
                                    step = i + 8*j
                                    if self.state[pos + step*k] == EMPTY:
                                        actions.append(
                                            Action(pos, pos + step*k, EMPTY))
                                        continue
                                    elif BLACK_MIN <= self.state[pos + step*k]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + step*k, EMPTY))
                                        break
                                    break
                                else:
                                    break

                    elif unit == WHITE_KING:
                        # King moves
                        for (i, j) in ((1, 0), (1, 1), (0, 1), (-1, 1),
                                       (-1, 0), (-1, -1), (0, -1), (1, -1)):
                            in_bounds = ((0 <= row + j < 8) and
                                         (0 <= col + i < 8))

                            target = self.state[pos + i + 8*j]
                            if in_bounds and (target == EMPTY or
                                              BLACK_MIN <= target < BLACK_MAX):
                                actions.append(
                                        Action(pos, pos + i + 8*j, EMPTY))

                        if (self.state[-1] and
                                self.state[5] ==
                                self.state[6] == EMPTY):
                            # King-side castling
                            actions.append(Action(4, 7))

                        if (self.state[-1] and
                                self.state[1] ==
                                self.state[2] ==
                                self.state[3] == EMPTY):
                            # Queen-side castling
                            actions.append(Action(4, 0))
                    else:
                        print("Unexpected unit")
                        print(unit)

        return actions

    def flip_board(self) -> "State":
        new_state = [EMPTY]*64 + self.state[-4:]
        for row in range(8):
            for col in range(8):
                new_state[col + 8*row] = 16 - self.state[col +8*(7-row)]
        return State(new_state)


class Action:
    def __init__(self, from_square: int, to_square: int, promotion):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion

    def __str__(self):
        from_row = self.from_square // 8
        from_col = self.from_square % 8

        to_row = self.to_square // 8
        to_col = self.to_square % 8
        return f"{from_row}-{from_col}--{to_row}-{to_col}--{self.promotion}"

    def __repr__(self):
        return self.__str__()


a = State(INIT_STATE)
b = a.get_actions()
