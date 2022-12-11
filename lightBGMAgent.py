import game
import random
import sklearn
import numpy as np
import joblib
import lightgbm as lgb

class agent():
    def __init__(self, model = lgb.LGBMRegressor(), pretrain_model = None, training_games = 100, machine_numbers = 50, game_rounds = 1000, train = True, save_model = 'lgb_model'):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = model
        self.prob = [ 0 for i in range(machine_numbers)]
        self.save_model = save_model
        if pretrain_model:
            print('Load pretrain model:', pretrain_model)
            self.model = joblib.load(pretrain_model)
        if train and training_games:
            self.train(training_games)
    def train(self, training_games = 100):
        train_X = []
        train_Y = []
        for i in range(training_games):
            print('traning steps', str(i+1) + '/' + str(training_games))
            G = game.game(self.machine_numbers, self.game_rounds, i)
            round = 0
            while round < self.game_rounds:
                if round < self.machine_numbers * 2:
                    if round % 2 == 0: #agent1
                        G.agent1Play(round // 2, False)
                    else:
                        G.agent2Play(round // 2, False)
                    round += 1
                    continue
                for j in range(self.machine_numbers):
                    if round % 2 == 0: #agent1
                        train_X.append([G.agent1Push[j] / round * 2, G.agent2Push[j] / round * 2, G.agent1MachineReward[j] / G.agent1Push[j]])
                        reward = G.agent1Play(j, False)
                        train_Y.append([G.machine[j]])
                        G.undoAgent1()
                    else:
                        train_X.append([G.agent2Push[j] / round * 2, G.agent1Push[j] / round * 2, G.agent2MachineReward[j] / G.agent2Push[j]])
                        reward = G.agent2Play(j, False)
                        train_Y.append([G.machine[j]])
                        G.undoAgent2()

                choice = int(random.random() * self.machine_numbers)
                if round % 2 == 0: #agent1
                    G.agent1Play(choice, False)
                else:
                    G.agent2Play(choice, False)
                round += 1
        train_Y = np.array(train_Y)
        train_Y = train_Y.reshape((-1,))
        self.model.fit(train_X, train_Y)
        if self.save_model:
            joblib.dump(self.model, self.save_model)
        print('counting scores')
        print(self.model.score(train_X, train_Y))
    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute):
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = []
        for i in range(machine_numbers):
            X.append([my_push_distribute[i] / current_round * 2, opp_push_distribute[i] / current_round * 2, my_reward_distribute[i] / my_push_distribute[i]])
        y = self.model.predict(X)
        # print(y.shape)
        for i in range(len(y)):
            y[i] *= 0.97 ** (my_push_distribute[i] + opp_push_distribute[i])
            self.prob[i] = y[i]
        maxProfit = -1
        choice = -1
        if agent == -1:
            for i in range(machine_numbers):
                if y[i][0] > maxProfit:
                    maxProfit = y[i][0]
                    choice = i
        else:
            for i in range(machine_numbers):
                if y[i] > maxProfit:
                    maxProfit = y[i]
                    choice = i
        return choice

if __name__ == '__main__':
    bot = agent()