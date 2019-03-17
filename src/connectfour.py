import numpy as np
from typing import List

import keras.backend as K
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation

symbols = {0: " ", 1: "■", -1: "□"}


class State:
    def __init__(self, state):
        self.state = state

    def __str__(self):
        output = "\n"
        for row in range(5, -1, -1):
            output = output + "|"
            for col in range(7):
                output = output + f"{symbols[self.state[6*col + row]]}|"
            output = output + "\n"
        return output

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "State") -> bool:
        if other is None:
            return False
        return self.state == other.state

    def play(self, action: "Action") -> "State":
        """Apply action to current state
        Returns next state, from the perspective of the other player.
        """
        new_state = self.state.copy()
        for row in range(6):
            if new_state[6*action.col + row] == 0:
                new_state[6*action.col + row] = 1
                break
        return State(new_state)

    def get_actions(self) -> List["Action"]:
        """Returns all the legal actions from the current state,
        from the perspective of White."""
        actions = []
        for col in range(7):
            for row in range(6):
                if self.state[6*col + row] == 0:
                    actions.append(Action(col))
                    break
        return actions

    def flip_board(self) -> "State":
        return State([-s for s in self.state])

    def is_terminal(self) -> bool:
        """Checks whether the game is over.
        Returns:
            - The truth value of whether it's over
            - +1 if '■' wins, -1 if '□' wins, 0 in case of a draw
        """
        s = self.state

        for value in (-1, 1):
            # Check vertical sequences
            for col in range(7):
                for row in range(3):
                    if s[6*col + row] == s[6*col + row+1] == s[6*col + row+2]\
                            == s[6*col + row+3] == value:
                        return (True, value)

            # Check horizontal sequences
            for row in range(6):
                for col in range(4):
                    if s[6*col + row] ==\
                            s[6*col + row + 6] ==\
                            s[6*col + row + 12] ==\
                            s[6*col + row + 18] == value:
                        return (True, value)

            # Check bottom-left top-right diagonals
            for col in range(4):
                for row in range(3):
                    if s[6*col + row] ==\
                            s[6*col + row + 7] ==\
                            s[6*col + row + 14] ==\
                            s[6*col + row + 21] == value:
                        return (True, value)

            # Check top-left bottom-right diagonals
            for col in range(4):
                for row in range(3):
                    if s[6*col + row + 3] ==\
                            s[6*col + row + 8] ==\
                            s[6*col + row + 13] ==\
                            s[6*col + row + 18] == value:
                        return (True, value)
        if 0 in s:
            return (False, 0)
        return (True, 0)

    def as_input(self) -> np.ndarray:
        """Returns a 0-1 valued vector representing the state of the board.
        With the same size as the input dimension of the neural network.
        """
        output = np.zeros(98, dtype=np.uint8)
        s = self.state
        for i in range(42):
            if s[i] == 1:
                output[2*i] = 1
            elif s[i] == -1:
                output[2*i + 1] = 1
        return output


class Action:
    def __init__(self, col):
        self.col = col

    def get_input():
        col = int(input("Col: "))
        return Action(col)

    def __str__(self):
        return str(self.col)

    def __eq__(self, other):
        if other is None:
            return False
        return self.col == other.col

    def __repr__(self):
        return self.__str__()

    def index(self):
        """Returns the index of the action, contained between 0 and the output
        dimension of the neural network.
        """
        return self.col


INIT_STATE = [0]*42

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
input_size = 98
output_size = 1 + 7

model = Sequential()
model.add(Dense(2048, input_dim=input_size,))
model.add(Activation('relu'))

model.add(Dense(output_size))
model.add(Activation('softmax'))
model.compile(loss=loss, optimizer='adam')

old_model = clone_model(model)
old_model.set_weights(model.get_weights())
