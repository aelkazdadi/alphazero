import numpy as np
from typing import List
from chess_constants import INIT_STATE, symbols
from chess_constants import EMPTY, WHITE_KING,\
    WHITE_BISHOP,\
    WHITE_QUEEN, WHITE_KNIGHT, WHITE_ROOK,\
    WHITE_PAWN, WHITE_PAWN_EN_PASSANT,\
    BLACK_PAWN, BLACK_PAWN_EN_PASSANT,\
    WHITE_MIN, WHITE_MAX,\
    BLACK_MIN, BLACK_MAX
import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation


class State:
    def __init__(self, state_vector: List):
        self.s: List[int]
        self.s = state_vector

    def __str__(self) -> str:
        string = "\n+---+---+---+---+---+---+---+---+\n"
        for row in range(7, -1, -1):
            string = string + "| "
            for column in range(8):
                piece = self.s[column + 8*row]
                string = string + symbols[piece] + " | "
            string = string + "\n+---+---+---+---+---+---+---+---+\n"
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: "State") -> bool:
        return self.s == other.s

    def play(self, action: "Action") -> "State":
        """Apply action to current state
        Returns next state, from the perspective of the other player.
        """
        from_square = action.from_square
        to_square = action.to_square

        new_state = self.s.copy()

        # Disallow castling
        if self.s[from_square] == WHITE_KING:
            new_state[-1] = False
            new_state[-2] = False

        elif self.s[from_square] == WHITE_ROOK:
            if from_square == 0:
                new_state[-1] = False
            elif to_square == 7:
                new_state[-2] = False

        if self.s[from_square] == WHITE_KING and\
                self.s[to_square] == WHITE_ROOK:
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

        elif (self.s[from_square] == WHITE_PAWN and
                8 <= from_square < 16 and
                to_square == from_square + 16):
            # En passant
            new_state[from_square] = EMPTY
            new_state[to_square] = WHITE_PAWN_EN_PASSANT

        elif (self.s[to_square] == EMPTY and
                self.s[from_square] == WHITE_PAWN and
                self.s[to_square - 8] == BLACK_PAWN_EN_PASSANT):
            # En passant capture
            new_state[from_square] = EMPTY
            new_state[to_square] = WHITE_PAWN

        else:
            # Standard move
            new_state[to_square] = self.s[from_square]
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
                piece = self.s[pos]

                if WHITE_MIN <= piece < WHITE_MAX:
                    # Pawn moves
                    if piece == WHITE_PAWN:
                        if self.s[pos+8] == EMPTY:
                            if 48 <= pos < 56:
                                # Promotion move
                                actions.append(Action(pos, pos+8, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+8, WHITE_KNIGHT))

                            else:
                                if 8 <= pos < 16:
                                    # First move
                                    actions.append(
                                        Action(pos, pos+16, WHITE_QUEEN))
                                # Standard move
                                actions.append(Action(pos, pos+8, WHITE_QUEEN))

                        if (BLACK_MIN <= self.s[pos+9] < BLACK_MAX or
                            self.s[pos+1] == BLACK_PAWN_EN_PASSANT) and\
                                (pos % 8) != 7:
                            # Capture
                            if 48 <= pos < 56:
                                # Capture and promotion
                                actions.append(Action(pos, pos+9, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+9, WHITE_KNIGHT))
                            else:
                                # Standard capture
                                actions.append(Action(pos, pos+9, WHITE_QUEEN))

                        if (BLACK_MIN <= self.s[pos+9] < BLACK_MAX or
                            self.s[pos-1] == BLACK_PAWN_EN_PASSANT) and\
                                (pos % 8) != 0:
                            if 48 <= pos < 56:
                                # Capture and promotion
                                actions.append(Action(pos, pos+7, WHITE_QUEEN))
                                actions.append(
                                    Action(pos, pos+7, WHITE_KNIGHT))
                            else:
                                # Standard capture
                                actions.append(Action(pos, pos+7, WHITE_QUEEN))
                    elif piece == WHITE_ROOK:
                        # Rook moves
                        for k in (-1, 1, -8, 8):
                            i = 0
                            while True:
                                i += 1
                                if k == -1 or k == 1:
                                    in_bounds = (8*row <= pos + k*i
                                                 < 8*(row+1))
                                else:
                                    in_bounds = (0 <= pos + k*i < 64)

                                if in_bounds:
                                    if self.s[pos + k*i] == EMPTY:
                                        actions.append(
                                            Action(pos, pos + k*i,
                                                   WHITE_QUEEN))
                                        continue
                                    elif BLACK_MIN <= self.s[pos + k*i]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + k*i,
                                                   WHITE_QUEEN))
                                        break
                                    break
                                else:
                                    break
                    elif piece == WHITE_BISHOP:
                        # Bishop moves
                        for k in ((-1, -8), (-1, 8), (1, -8), (1, 8)):
                            i = 0
                            while True:
                                i += 1
                                in_bounds = (0 <= pos + k[1]*i < 64) and\
                                    (8*row <= pos + k[0]*i < 8*(row+1))

                                if in_bounds:
                                    step = k[0] + k[1]
                                    if self.s[pos + step*i] == WHITE_QUEEN:
                                        actions.append(
                                            Action(pos, pos + step*i,
                                                   WHITE_QUEEN))
                                        continue
                                    elif BLACK_MIN <= self.s[pos + step*i]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + step*i,
                                                   WHITE_QUEEN))
                                        break
                                    break
                                else:
                                    break
                    elif piece == WHITE_KNIGHT:
                        # Knight moves
                        for (i, j) in ((1, 2), (2, 1), (-1, 2), (-2, 1),
                                       (-2, -1), (-1, -2), (2, -1), (1, -2)):
                            in_bounds = (0 <= row + j < 8) and\
                                (0 <= col + i < 8)
                            if in_bounds:
                                target = self.s[pos + i + 8*j]
                                if (target == EMPTY or
                                        BLACK_MIN <= target < BLACK_MAX):
                                    actions.append(
                                        Action(pos, pos + i + 8*j,
                                               WHITE_QUEEN))

                    elif piece == WHITE_QUEEN:
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
                                    if self.s[pos + step*k] == EMPTY:
                                        actions.append(
                                            Action(pos, pos + step*k,
                                                   WHITE_QUEEN))
                                        continue
                                    elif BLACK_MIN <= self.s[pos + step*k]\
                                            < BLACK_MAX:
                                        actions.append(
                                            Action(pos, pos + step*k,
                                                   WHITE_QUEEN))
                                        break
                                    break
                                else:
                                    break

                    elif piece == WHITE_KING:
                        # King moves
                        for (i, j) in ((1, 0), (1, 1), (0, 1), (-1, 1),
                                       (-1, 0), (-1, -1), (0, -1), (1, -1)):
                            in_bounds = ((0 <= row + j < 8) and
                                         (0 <= col + i < 8))

                            if in_bounds:
                                target = self.s[pos + i + 8*j]
                                if (target == EMPTY or
                                        BLACK_MIN <= target < BLACK_MAX):
                                    actions.append(
                                        Action(pos, pos + i + 8*j,
                                               WHITE_QUEEN))

                        if (self.s[-1] and
                                self.s[5] ==
                                self.s[6] == EMPTY):
                            # King-side castling
                            actions.append(Action(4, 7, WHITE_QUEEN))

                        if (self.s[-1] and
                                self.s[1] ==
                                self.s[2] ==
                                self.s[3] == EMPTY):
                            # Queen-side castling
                            actions.append(Action(4, 0, WHITE_QUEEN))
                    else:
                        print("Unexpected piece")
                        print(piece)

        return actions

    def flip_board(self) -> "State":
        new_state = [EMPTY]*64 + self.s[-4:]
        for row in range(8):
            for col in range(8):
                new_state[col + 8*row] = 16 - self.s[col + 8*(7-row)]
        return State(new_state)

    def is_terminal(self) -> bool:
        for i in range(64):
            if self.s[i] == WHITE_KING:
                return (False,)
        return (True, -1)

    def as_input(self) -> np.ndarray:
        v = np.zeros((64, 9), dtype=np.uint8)
        for i in range(64):
            piece = self.s[i]
            if piece != EMPTY:
                if WHITE_MIN <= piece < WHITE_MAX:
                    v[i][0] = 1
                    v[i][piece+2] = 1
                else:
                    v[i][1] = 1
                    v[i][16-piece + 2] = 1
        return np.concatenate((v.flatten(),
                               np.array(self.s[-4:], dtype=np.uint8)))


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

    def flip_action(self):
        from_row = self.from_square // 8
        from_col = self.from_square % 8

        to_row = self.to_square // 8
        to_col = self.to_square % 8
        return Action(from_col + 8*(7-from_row),
                      to_col + 8*(7-to_row), self.promotion)

    def index(self):
        promotion = 0 if self.promotion == WHITE_QUEEN else 1
        return self.from_square + 64*self.to_square\
            + 4096*promotion


initial_state = State(INIT_STATE)
white_header = "+---------------+\n| White's turn  |"
black_header = "+---------------+\n| Black's turn  |"


def proba_list(state: State, actions: List[Action], dense_probas: np.ndarray):
    probas = []
    for action in actions:
        proba = dense_probas[action.index()]
        probas.append(proba)
    total = sum(probas)
    for i in range(len(probas)):
        probas[i] /= total
    return probas


def value_prior(state: State, actions: List[Action], neural_network):
    nn_output = neural_network.predict(
        state.as_input()[np.newaxis, :])[0]
    value = 2*nn_output[0] - 1
    prior = proba_list(state, actions, nn_output[1:])
    return value, prior


def loss(y_true, y_pred):
    return (K.sum(
        (y_true[:, 0] - y_pred[:, 0])**2
        - K.dot(y_true[:, 1:], K.log(y_pred[:, 1:]))))


input_size = 580
output_size = 8193
C2 = 0.01

model = Sequential()
model.add(Dense(output_size, input_shape=(input_size,),
                kernel_regularizer=regularizers.l2(C2),
                bias_regularizer=regularizers.l2(C2)))
model.add(Activation('softmax'))
model.compile(loss=loss, optimizer='adam')

old_model = Sequential()
old_model.add(Dense(output_size, input_shape=(input_size,),
                    kernel_regularizer=regularizers.l2(C2),
                    bias_regularizer=regularizers.l2(C2)))
old_model.add(Activation('softmax'))
old_model.compile(loss=loss, optimizer='adam')
