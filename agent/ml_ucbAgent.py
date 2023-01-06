import numpy as np
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import math


class agent():
    def __init__(self, modelClass=KNeighborsRegressor(), machine_numbers=50, game_rounds=1000, train=True, save_model='lgb_model'):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = joblib.load(modelClass.__class__.__name__+'.h5')
        self.c = math.sqrt(2)
        self.adaptive = True

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
        agent1AdjustPush = data.adjustedChoose
        agent2AdjustPush = data.opponentAdjustedChoose
        agent1AdjustMachineReward = data.adjustedSuccessTime
        if len(my_history_choice) < machine_numbers:  # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = []
        for i in range(machine_numbers):
            X.append([
                np.array(len(my_history_choice)),
                np.array(my_push_distribute[i]),
                np.array(agent1AdjustPush[i]),
                np.array(my_reward_distribute[i]),
                np.array(agent1AdjustMachineReward[i]),
                np.array(
                    my_reward_distribute[i]/my_push_distribute[i] if my_push_distribute[i] != 0 else 0),
                np.array(
                    agent1AdjustMachineReward[i]/agent1AdjustPush[i] if agent1AdjustPush[i] != 0 else 0),
                np.array(opp_push_distribute[i]),
                np.array(agent2AdjustPush[i])
            ])
        X = np.array(X)
        y = self.model.predict(X)
        
        N = len(my_history_choice)
        n = [0 for i in range(machine_numbers)]
        for i in range(len(my_history_choice)):
            n[my_history_choice[i]] += 1

        index = -1
        maxUCB = -1e9
        index2 = -1
        maxY = -1e9
        for i in range(len(y)):
            ucb = y[i] + self.c * math.sqrt(math.log(N) / n[i])
            if ucb > maxUCB:
                maxUCB = ucb
                index = i
            if y[i] > maxY:
                maxY = y[i]
                index2 = i
        if self.adaptive:
            if index == index2:
                self.c *= 1.1
            else:
                self.c *= 0.9
        return index


if __name__ == '__main__':
    bot = agent()
