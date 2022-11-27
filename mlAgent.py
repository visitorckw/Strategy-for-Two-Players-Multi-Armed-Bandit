import game
import random
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

class agent():
    def __init__(self, machine_numbers = 50, game_rounds = 1000, model = LinearRegression(), train = True, training_games = 100):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = model
        if train:
            self.train(training_games)
    def train(self, training_games = 100):
        train_X = []
        train_Y = []
        for i in range(training_games):
            print('traning steps', str(i+1) + '/' + str(training_games))
            G = game.game(self.machine_numbers, self.game_rounds, i)
            round = 0
            while round < self.game_rounds:
                if round < self.machine_numbers:
                    if round % 2 == 0: #agent1
                        G.agent1Play(round // 2, False)
                    else:
                        G.agent2Play(round // 2, False)
                    round += 1
                    continue
                for j in range(self.machine_numbers):
                    if round % 2 == 0: #agent1
                        reward = G.agent1Play(j, False)
                        train_X.append([G.agent1Push[j], G.agent2Push[j], G.agent1MachineReward[j]])
                        train_Y.append([reward])
                        G.undoAgent1()
                    else:
                        reward = G.agent2Play(j, False)
                        train_X.append([G.agent2Push[j], G.agent1Push[j], G.agent2MachineReward[j]])
                        train_Y.append([reward])
                        G.undoAgent2()

                choice = int(random.random() * self.machine_numbers)
                if round % 2 == 0: #agent1
                    G.agent1Play(choice, False)
                else:
                    G.agent2Play(choice, False)
                round += 1
        self.model.fit(train_X, train_Y)
        # print(self.model.score(train_X, train_Y))
    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute):
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = []
        for i in range(machine_numbers):
            X.append([my_push_distribute[i], opp_push_distribute[i], my_reward_distribute[i]])
        y = self.model.predict(X)
        maxProfit = -1
        choice = -1
        if agent == 1 or agent == 2:
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