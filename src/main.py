import tictactoe as game
from tqdm import tqdm

import numpy as np
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

c_mcts = 3.
mcts_iter_default = 80

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
        """
        Basic tree class, used for the monte carlo tree search.
        Is instanciated with the root state.
        """
        self.state = state
        self.parent = parent
        self.actions = []
        self.children = []


def mcts(state, neural_network, mcts_iter=mcts_iter_default, C=c_mcts):
    tree = Tree(state)
    """
    Function that operates the monte carlo tree search
    starting from a given state.
    :param state: state from which to start the mcts.
        The class state is defined and documented in each game.
    :param neural_network: implementation used is Keras.
    :param mcts_iter: number of iterations.
    :param C: Exploration rate parameter.
    :return: np array, containing the improved policy.
    """
    for _ in range(mcts_iter):
        node = tree
        children = node.children
        # Traverse the tree until a leaf node is found.
        while children != []:
            sqrtN_tot = sqrt(sum([a.N for a in node.actions]))
            # Choose the child node with the largest PUCT
            puct_max_index = np.argmax([a.Q + C*a.P*sqrtN_tot/(1+a.N)
                                        for a in node.actions])

            node = children[puct_max_index]
            children = node.children

        # Create children of the leaf node if it's not terminal
        # Then propagate the values along the visited path of the tree
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
    # Improved policy is proportional to number of visits of each child node.
    policy = [action.N for action in node.actions]
    if policy == []:
        return []

    return np.array(policy)/sum(policy)


def one_episode(side, max_turns=10000, verbose=0,
                mcts_iter=mcts_iter_default, C=c_mcts,
                model=model, old_model=old_model,
                exploration=True):
    """
    Function that generates an episode of the game from a given perspective,
    defined here as side.
    :param side: -1:'black' or 1:'white'.
    :param max_turns:  maximum number of turns.
    :param verbose: amount of information to display.
        0  : Nothing is printed.
        1  : Outcome of the game is printed.
        2+ : Each turn of the game is additionally printed.
    :param mcts_iter: number of MCTS iterations.
    :param C: exploration parameters.
    :return: inputs:  Contains the visited states, in the form
                      of neural network inputs.
             ouptuts: Contains the outcome of the game,
                      and the improved policies, in the form
                      of neural network outputs.
    """
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
            # Model is playing
            policy = mcts(state, model, mcts_iter=mcts_iter, C=C)
        else:
            if verbose > 1:
                print(black_header, end='')
                print(state.flip_board())
            # Old model is playing
            policy = mcts(state, old_model, mcts_iter=mcts_iter, C=C)
        actions = state.get_actions()

        if exploration:
            # Choose a move from the distribution given by the MCTS
            action = np.random.choice(actions, p=policy)
        else:
            # Choose the best move given by the MCTS
            action = actions[np.argmax(policy)]

        if i % 2 == 0:
            # Update data with the current state and generated policy
            data.append([state, actions, policy, None])

        state = state.play(action).flip_board()
        result = state.is_terminal()
        game_over = result[0]
        i += 1

    if i == max_turns:
        return None, None

    # Update outcome of the game in the data
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

    # Transform the data into a sparse matrix
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

    outputs = csr_matrix((dat, indices, indptr),
                         shape=(len(data), output_size))
    return inputs, outputs


def dataset(n=200, verbose=0, mcts_iter=mcts_iter_default, C=c_mcts,
            model=model, old_model=old_model, exploration=True):
    """
    Generate a dataset by repeatedly pitting
    two models against each other.
    :param n: number of episodes to be simulated
    """
    # Initialize empty dataset
    inputs = np.zeros((0, input_size))
    outputs = csr_matrix((0, output_size))

    win = 0
    loss = 0
    draw = 0
    for _ in tqdm(range(n), ncols=0):
        side = np.random.choice([1, -1])
        # Play one episode as either the first or second player.
        i, o = one_episode(side, mcts_iter=mcts_iter, C=C, verbose=verbose,
                           model=model, old_model=old_model,
                           exploration=exploration)

        # Append inputs and outputs to dataset
        if i is not None and o is not None:
            inputs = np.vstack((inputs, i))
            outputs = vstack((outputs, o))

        # Update stats so far.
        if o[0, 0] == 1.:
            win += 1
        elif o[0, 0] == 0.:
            loss += 1
        else:
            draw += 1
    return inputs, outputs, (win/n, draw/n, loss/n)


def play_bot(state, model, mcts_iter=mcts_iter_default, C=0.):
    """
    Esimtate the best action from a given state,
    according to the MCTS generated by a given neural network.
    """
    actions = state.get_actions()
    policy = mcts(state, model, mcts_iter=mcts_iter, C=C)
    action = actions[np.argmax(policy)]
    return state.play(action)


def play_game(state=initial_state, model=model,
              mcts_iter=mcts_iter_default, C=0.):
    """
    Play a game against a neural netork model.
    """
    result = state.is_terminal()
    while not result[0]:
        print(state)
        action = None
        legal_actions = state.get_actions()
        while not (action in legal_actions):
            action = Action.get_input()
        state = state.play(action).flip_board()
        result = state.is_terminal()

        if result[0]:
            print()
            if result[1] == 0:
                print("Draw")
            elif result[1] == -1:
                print("Win")
            elif result[1] == 1:
                print("Loss")
            return

        state = play_bot(state, model, mcts_iter=mcts_iter, C=C).flip_board()
        result = state.is_terminal()

    print()
    if result[1] == 0:
        print("Draw")
    elif result[1] == 1:
        print("Win")
    elif result[1] == -1:
        print("Loss")
    return


def train(train_iter=20, model=model, old_model=old_model, n_episodes=100,
          train_epochs=100, mcts_iter=mcts_iter_default, C=c_mcts,
          winrate_threshold=.55, disp_winrate=False):
    """
    Generate a dataset and train the neural
    network on the obtained output, while updating the opponent
    neural network once the model to be trained has sufficiently improved.

    :param train_iter: Number of training iterations.
    :param train_epochs: Neural network fitting iteration parameter
    """
    full_stats = []
    for i in range(train_iter):
        inputs, outputs, stats = dataset(n_episodes, verbose=0,
                                         mcts_iter=mcts_iter, C=C,
                                         model=model,
                                         old_model=old_model)
        model.fit(inputs, outputs, epochs=train_epochs, verbose=0)
        full_stats.append(stats)

        winrate = stats[0]/(stats[0] + stats[2])

        if disp_winrate:
            print("Win/Draw/Loss (%) :", stats)
            print("Winrate :", winrate)
        if winrate > winrate_threshold:
            print("Old model updated")
            old_model.set_weights(model.get_weights())
            full_stats.append((0., 0., 0.))

    return full_stats
