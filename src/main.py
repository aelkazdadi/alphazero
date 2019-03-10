import numpy as np
import chess
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation


C1 = 10.
C2 = 0.01


def loss(y_true, y_pred):
    return (K.sum(
        (y_true[:, 0] - y_pred[:, 0])**2
        - K.dot(y_true[:, 1:], K.log(y_pred[:, 1:]))))


model = Sequential()
model.add(Dense(16385, input_shape=(512,),
                kernel_regularizer=regularizers.l2(C2),
                bias_regularizer=regularizers.l2(C2)))
model.add(Activation('softmax'))
model.compile(loss=loss, optimizer='adam')

old_model = Sequential()
old_model.add(Dense(16385, input_shape=(512,),
                    kernel_regularizer=regularizers.l2(C2),
                    bias_regularizer=regularizers.l2(C2)))
old_model.add(Activation('softmax'))
old_model.compile(loss=loss, optimizer='adam')


def initial_state():
    return State(chess.Board())


class State:
    def __init__(self, s):
        self.s = s

    def play(self, action):
        self.s.push(action.a)
        copy = self.s.copy()
        self.s.pop()
        return State(copy)

    def flip_board(self):
        return State(self.s.mirror())

    def get_actions(self):
        actions = list(self.s.legal_moves)
        return [Action(a) for a in actions]

    def is_terminal(self):
        is_over = self.s.is_game_over()
        if is_over:
            result = self.s.result
            if result == '1-0':
                return (True, 1)
            elif result == '0-1':
                return (True, -1)
            else:
                return (True, 0)
        else:
            return (False, 0)

    def as_input(self):
        v = np.zeros((64, 8))
        for i in range(64):
            piece = self.s.piece_at(i)
            piece_type = self.s.piece_type_at(i)
            if piece is not None:
                v[i][0] = int(piece.color)
                v[i][1] = 1 - int(piece.color)
                for j in range(2, 8):
                    v[i][j] = int(piece_type+1 == j)
        return v.flatten()


class Action:
    def __init__(self, a):
        self.a = a

    def index(self):
        promotion = (self.a.promotion or 2) - 2
        return self.a.from_square + 64*self.a.to_square + 4096*promotion


class Tree:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.actions = []
        self.children = []


def proba_list(state, actions, dense_probas):
    probas = []
    for action in actions:
        proba = dense_probas[action.index()]
        probas.append(proba)
    return probas


def mcts(state, neural_network, n_iter=500):
    tree = Tree(state)

    for _ in range(n_iter):
        node = tree
        children = node.children
        while children != []:
            ucb_index = np.argmax([a.Q + C1*a.P/(1+a.N) for a in node.actions])

            node = children[ucb_index]
            children = node.children

        terminal = node.state.is_terminal()
        if terminal[0]:
            value = terminal[1]
        else:
            node.actions = node.state.get_actions()
            node.children = [Tree(node.state.play(a).flip_board(),
                                  parent=(node, a)) for a in node.actions]

            nn_output = neural_network.predict(
                node.state.as_input()[np.newaxis, :])[0]
            value = 2*nn_output[0] - 1
            prior = proba_list(node.state, node.actions, nn_output[1:])

            for (i, a) in enumerate(node.actions):
                a.P = prior[i]
                a.N = 0
                a.Q = 0
        while node.parent is not None:
            value = -value
            N = node.parent[1].N
            Q = node.parent[1].Q
            node.parent[1].Q = N/(N+1) * Q + value/(N+1)
            node.parent[1].N += 1

            node = node.parent[0]
    policy = [action.N for action in node.actions]
    if policy == []:
        return []
    return np.array(policy)/sum(policy)


def one_episode(side, max_turns=10000):
    data = []
    state = initial_state()
    if side == -1:
        state = state.flip_board()
    game_over = False

    i = 0
    while not game_over and i < max_turns:
        if i % 2 == 0:
            print(state.s)
        else:
            print(state.flip_board().s)
        print()
        if i % 2 == 0:
            policy = mcts(state, model)
        else:
            policy = mcts(state, old_model)
        actions = state.get_actions()
        action = np.random.choice(actions, p=policy)

        if i % 2 == 0:
            data.append([state, actions, policy, None])

        state = state.play(action).flip_board()
        result = state.is_terminal()
        game_over = result[0]
        i += 1

    if i == max_turns:
        return None, None

    result = result[1]
    for i in range(len(data)):
        result = -result
        data[0-i][-1] = (result+1)//2

    inputs = np.zeros((len(data), 512))
    dat = []
    indices = []
    indptr = [0]
    for datum in data:
        inputs[i] = datum[0].as_input()

        indices.append(0)
        indices.extend([action.index() for action in datum[1]])

        indptr.append(len(datum[1])+1)

        dat.append(datum[3])
        dat.extend(datum[2])

    outputs = csr_matrix((dat, indices, indptr), shape=(len(data), 16385))
    return inputs, outputs


def dataset(n=200):
    inputs = np.zeros((0, 512))
    outputs = csr_matrix((0, 16385))
    for _ in range(n):
        i, o = one_episode(side=np.random.choice([-1, 1]))
        if i is not None and o is not None:
            inputs = np.vstack(inputs, i)
            outputs = vstack(outputs, o)
