from time import time
from main import model, old_model,\
    train, dataset
from keras.models import clone_model

model = model
old_model = old_model

first_model = clone_model(model)
first_model.set_weights(model.get_weights())

models = []

params = [(500, 100),
          (500, 50),
          (500, 25),
          (1000, 100),
          (2000, 100),
          ]

training_stats = []
testing_stats = []

for n, mcts_iter in params:
    begin = time()
    print("Params :", n, mcts_iter)
    model.set_weights(first_model.get_weights())
    old_model.set_weights(first_model.get_weights())

    training_stats.append(train(train_iter=20, model=model,
                                old_model=old_model, n_episodes=n,
                                mcts_iter=mcts_iter, C=1.,
                                winrate_threshold=.55,
                                train_epochs=200, disp_winrate=True))

    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())
    models.append(new_model)

    end = time()
    print(f"Training took {end-begin} seconds.")

for m in models:
    _, _, tmp = dataset(1000, verbose=0, mcts_iter=10, C=1.,
                        exploration=False, model=m,
                        old_model=first_model)
    testing_stats.append(tmp)
