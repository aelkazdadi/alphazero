import tictactoe as game

import numpy as np
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

C = 1.

State = game.State
Action = game.Action
initial_state = game.initial_state
white_header = game.white_header
black_header = game.black_header

input_size = game.input_size
output_size = game.output_size
model = game.model
old_model = game.old_model

value_prior = game.value_prior


class Tree:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.actions = []
        self.children = []


def mcts(state, neural_network, n_iter=100, debug=False, C=C):
    tree = Tree(state)

    for _ in range(n_iter):
        node = tree
        children = node.children
        while children != []:
            sqrtN_tot = sqrt(sum([a.N for a in node.actions]))
            ucb_index = np.argmax([a.Q + C*a.P*sqrtN_tot/(1+a.N)
                                   for a in node.actions])

            node = children[ucb_index]
            children = node.children

        terminal = node.state.is_terminal()
        if terminal[0]:
            value = terminal[1]

        else:
            node.actions = node.state.get_actions()
            node.children = [Tree(node.state.play(a).flip_board(),
                                  parent=(node, a)) for a in node.actions]

            value, prior = value_prior(node.state, node.actions,
                                            neural_network)

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

    if debug:
        for action in node.actions:
            Q = "{:.3e}".format(action.Q)
            P = "{:.3e}".format(action.P)
            if Q[0] != "-":
                Q = "+"+Q
            if P[0] != "-":
                P = "+"+P
            print("{:<5}".format(action.N),
                  "{:<10}".format(Q),
                  "{:<10}".format(P),
                  action)

    return np.array(policy)/sum(policy)


def one_episode(side, max_turns=10000, verbose=0, C=C):
    data = []
    state = initial_state

    i = 0
    if side == -1:
        state = state.flip_board()
        i += 1
    game_over = False

    while not game_over and i < max_turns:
        if i % 2 == 0:
            if verbose > 1:
                print(white_header, end='')
                print(state)
            policy = mcts(state, model, C=C)
        else:
            if verbose > 1:
                print(black_header, end='')
                print(state.flip_board())
            policy = mcts(state, old_model, C=C)
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

    if result[1] == 0:
        result = .5
        if verbose > 1:
            if i % 2 == 0:
                print(state)
            else:
                print(state.flip_board())
        if verbose > 0:
            print("Draw")
    else:
        result = (i + (result[1]+1)//2) % 2
        if i % 2 == 0:
            if verbose > 1:
                print(state)
            if verbose > 0:
                print("Black victory")
        else:
            if verbose > 1:
                print(state.flip_board())
            if verbose > 0:
                print("White victory")
    for i in range(len(data)):
        data[i][3] = result

    inputs = np.zeros((len(data), input_size))
    dat = []
    indices = []
    indptr = [0]
    for (i, datum) in enumerate(data):
        inputs[i] = datum[0].as_input()

        indices.append(0)
        indices.extend([1 + action.index() for action in datum[1]])

        indptr.append(indptr[-1] + len(datum[1])+1)

        dat.append(datum[3])
        dat.extend(datum[2])

    outputs = csr_matrix((dat, indices, indptr), shape=(len(data), output_size))
    return inputs, outputs


winrate = .5


def dataset(n=200, verbose=0, C=C):
    inputs = np.zeros((0, input_size))
    outputs = csr_matrix((0, output_size))

    win = 0
    win_loss = 0
    for _ in range(n):
        side = np.random.choice([1, -1])
        if verbose > 0:
            print(side)
        i, o = one_episode(side, verbose=verbose)
        if i is not None and o is not None:
            inputs = np.vstack((inputs, i))
            outputs = vstack((outputs, o))

        if o[0, 0] == 1.:
            win += 1
        if o[0, 0] != .5:
            win_loss += 1
    global winrate
    winrate = win/win_loss
    return inputs, outputs


def compare_models():
    b = True
    for i in range(3):
        b = b and (model.get_weights()[i] == old_model.get_weights()[i]).all()
    print(b)


def play_bot(state, model, C=C):
    state = state.flip_board()
    actions = state.get_actions()
    policy = mcts(state, model, C=C)
    action = actions[np.argmax(policy)]
    return state.play(action).flip_board()


n = 500

for i in range(30):
    compare_models()
    res = dataset(n, verbose=0, C= 1.)
    model.fit(res[0], res[1], epochs=100, verbose=0)
    if winrate > .6:
        old_model.set_weights(model.get_weights())
    print("Winrate:", winrate)

print()
res = dataset(4*n, verbose=0, C=0)
print("Winrate:", winrate)
