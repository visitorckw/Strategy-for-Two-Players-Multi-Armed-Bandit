# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import random
import game

class agent():
    def __init__(self, pretrain_model = None, training_games = 100, machine_numbers = 50, game_rounds = 1000, train = True):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.prob = [0 for i in range(machine_numbers)]
        
        # Initialising the RNN
        self.model = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.machine_numbers, 3)))
        # regressor.add(LSTM(units = 50, return_sequences = True))
        # self.model.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True))
        # self.model.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True))
        # self.model.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50))
        # self.model.add(Dropout(0.2))

        # Adding the output layer
        self.model.add(Dense(units = 1))

        # Compiling
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        self.model.summary()

        if pretrain_model:
            self.model.load_weights('lstm_model.h5')
        if train and training_games:
            self.train(training_games)

    def train(self, training_games):
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
                        my_history_choice = G.historyAgent1Choice[-self.machine_numbers:]
                        opp_history_choice = G.historyAgent2Choice[-self.machine_numbers:]
                        my_history_reward = G.historyAgent1Reward[-self.machine_numbers:]
                        x = []
                        for k in range(self.machine_numbers):
                            my_choice = my_history_choice[k] == j
                            opp_choice = opp_history_choice[k] == j
                            my_reward = my_history_reward[k] if my_choice else 0
                            x.append([my_choice, opp_choice, my_reward])
                        train_X.append(x)
                        reward = G.agent1Play(j, False)
                        train_Y.append([reward])
                        G.undoAgent1()
                    else:
                        my_history_choice = G.historyAgent2Choice[-self.machine_numbers:]
                        opp_history_choice = G.historyAgent1Choice[-self.machine_numbers:]
                        my_history_reward = G.agent2MachineReward[-self.machine_numbers:]
                        x = []
                        for k in range(self.machine_numbers):
                            my_choice = my_history_choice[k] == j
                            opp_choice = opp_history_choice[k] == j
                            my_reward = my_history_reward[k] if my_choice else 0
                            x.append([my_choice, opp_choice, my_reward])
                        train_X.append(x)
                        reward = G.agent2Play(j, False)
                        train_Y.append([G.machine[j]])
                        G.undoAgent2()

                choice = int(random.random() * self.machine_numbers)
                if round % 2 == 0: #agent1
                    G.agent1Play(choice, False)
                else:
                    G.agent2Play(choice, False)
                round += 1
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        train_Y = train_Y.reshape((-1,))
        print(train_X.shape)
        print(train_Y.shape)
        self.model.fit(train_X, train_Y, epochs = 10, batch_size = 32)
        self.model.save_weights('../lstm_model.h5')

    def play(self, data):
        agent = data.agent
        machine_numbers = data.machine_numbers
        total_round = data.total_round
        current_round = data.current_round
        my_total_rewards = data.my_total_rewards
        my_history_choice = data.my_history_choice 
        opp_history_choice = data.opp_history_choice 
        my_history_reward = data.my_history_reward
        my_push_distribute = data.my_push_distribute
        opp_push_distribute = data.opp_push_distribute
        my_reward_distribute = data.my_reward_distribute
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        expectProfit = []
        for i in range(machine_numbers):
            my_history_choice = my_history_choice[-self.machine_numbers:]
            opp_history_choice = opp_history_choice[-self.machine_numbers:]
            my_history_reward = my_reward_distribute[-self.machine_numbers:]
            x = []
            for k in range(self.machine_numbers):
                my_choice = my_history_choice[k] == i
                opp_choice = opp_history_choice[k] == i
                my_reward = my_history_reward[k] if my_choice else 0
                x.append([my_choice, opp_choice, my_reward])
            y = self.model.predict([x])
            y[0][0] *= 0.97 ** (my_push_distribute[i] + opp_push_distribute[i])
            self.prob[i] = y[0][0]
            # print(y.shape)
            expectProfit.append(y[0][0])
        maxProfit = -1
        choice = -1
        for i in range(self.machine_numbers):
            if expectProfit[i] > maxProfit:
                maxProfit = expectProfit[i]
                choice = i
        print('lstm predict', expectProfit)
        return choice