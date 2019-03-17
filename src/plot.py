from time import time
from main import model, old_model,\
    train, dataset
from keras.models import clone_model
import numpy as np
import matplotlib.pyplot as plt

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

testing_stats = []
for m in models:
    _, _, tmp = dataset(100, verbose=0, mcts_iter=50, C=1.,
                        exploration=False, model=m,
                        old_model=first_model)
    testing_stats.append(tmp)

for i in range(len(models)):
    res = np.array(training_stats[i])
    ind = np.arange(res.shape[0])
    width = .8
    winrate = res[:, 0]/(res[:, 0] + res[:, 2])
    for j in range(winrate.size):
        if np.isnan(winrate[j]):
            winrate[j] = winrate[j-1]

    ticks = []
    count = 1
    for j in range(res.shape[0]):
        if (res[j] == np.zeros(3)).all():
            ticks.append('')
        else:
            ticks.append(str(count))
            count += 1

    p1 = plt.bar(ind, res[:, 0], width, color=np.array((55, 126, 184))/255)
    p2 = plt.bar(ind, res[:, 1], width, bottom=res[:, 0],
                 color=np.array((77, 175, 74))/255)
    p3 = plt.bar(ind, res[:, 2], width, bottom=res[:, 0] +
                 res[:, 1], color=np.array((228, 26, 28))/255)
    plt.plot(ind, winrate, linewidth=2., color='black')
    plt.hlines(.55, ind[0]-width/2, ind[-1]+width/2, linestyles='dashed')
    plt.tight_layout()
    plt.xlim(ind[0]-width/2, ind[-1]+width/2)
    plt.ylim(0, 1)
    plt.xticks(ind, ticks)
    plt.savefig(str(i)+".pdf")
    plt.show()
