import numpy as np
from typing import List

import keras.backend as K
from keras import regularizers
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation

symbols = {0: " ", 1: "O", -1: "X"}


class State:
    def __init__(self, state):
        self.state = state

    def __str__(self):
        output = "\n+---+---+---+\n"
        for row in range(3):
            output = output + "|"
            for col in range(3):
                output = output + f" {symbols[self.state[col + 3*row]]} |"
            output = output + "\n+---+---+---+\n"
        return output

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "State") -> bool:
        return self.state == other.state

    def play(self, action: "Action") -> "State":
        """Apply action to current state
        Returns next state, from the perspective of the other player.
        """
        new_state = self.state.copy()
        new_state[action.square] = 1
        return State(new_state)

    def get_actions(self) -> List["Action"]:
        """Returns all the legal actions from the current state,
        from the perspective of White."""
        actions = []
        for i in range(9):
            if self.state[i] == 0:
                actions.append(Action(i))
        return actions

    def flip_board(self) -> "State":
        return State([-s for s in self.state])

    def is_terminal(self) -> bool:
        """Checks whether the game is over.
        Returns:
            - The truth value of whether it's over
            - +1 if 'O' wins, -1 if 'X' wins, 0 in case of a draw
        """
        s = self.state

        diag1_result = s[0] + s[4] + s[8]
        diag2_result = s[2] + s[4] + s[6]

        if diag1_result == -3:
            return (True, -1)
        if diag2_result == -3:
            return (True, -1)

        if diag1_result == 3:
            return (True, 1)
        if diag2_result == 3:
            return (True, 1)

        for i in range(3):
            row_result = s[3*i]+s[1 + 3*i]+s[2 + 3*i]
            col_result = s[i]+s[i + 3]+s[i + 6]

            if row_result == -3:
                return (True, -1)
            if col_result == -3:
                return (True, -1)

            if row_result == 3:
                return (True, 1)
            if col_result == 3:
                return (True, 1)

        if not (0 in s):
            return (True, 0)
        return (False, 0)

    def as_input(self) -> np.ndarray:
        """Returns a 0-1 valued vector representing the state of the board.
        With the same size as the input dimension of the neural network.
        """
        output = np.zeros(18, dtype=np.uint8)
        s = self.state
        for i in range(9):
            if s[i] == 1:
                output[2*i] = 1
            elif s[i] == -1:
                output[2*i + 1] = 1
        return output


class Action:
    def __init__(self, square):
        self.square = square

    def __str__(self):
        col = self.square % 3
        row = (self.square)//3
        return f"{row}-{col}"

    def __repr__(self):
        return self.__str__()

    def index(self):
        """Returns the index of the action, contained between 0 and the output
        dimension of the neural network.
        """
        return self.square

INIT_STATE = [0]*9

initial_state = State(INIT_STATE)
white_header = "+---+\n| O |"
black_header = "+---+\n| X |"


def loss(y_true, y_pred):
    return (K.sum(
        (y_true[:, 0] - y_pred[:, 0])**2
        - K.dot(y_true[:, 1:], K.transpose(K.log(y_pred[:, 1:])))))


def proba_list(state: State, actions: List[Action], dense_probas: np.ndarray):
    """Returns the policy over legal actions, given the policy over all the
    actions.
        - actions: list of legal actions
        - dense_probas: policy given by the neural network
    Returns the policy over the legal actions.
    """
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


# Neural network architecture
input_size = 18
output_size = 10

model = Sequential()
model.add(Dense(256, input_dim=input_size,))
model.add(Activation('relu'))

model.add(Dense(output_size))
model.add(Activation('softmax'))
model.compile(loss=loss, optimizer='adam')

old_model = clone_model(model)
old_model.set_weights(model.get_weights())
