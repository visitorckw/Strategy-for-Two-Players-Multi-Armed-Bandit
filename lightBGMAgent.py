import game
import random
import sklearn
import numpy as np
import joblib
import lightgbm as lgb
import pandas as pd
class agent():
    def __init__(self, modelFile = "lgb_model", pretrain_model = None, training_games = 100, machine_numbers = 50, game_rounds = 1000, train = True, save_model = 'lgb_model'):
        self.machine_numbers = machine_numbers
        self.game_rounds = game_rounds
        self.model = lgb.Booster(model_file=modelFile)
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
        agent1AdjustPush= data.adjustedChoose
        agent2AdjustPush = data.opponentAdjustedChoose
        agent1AdjustMachineReward = data.adjustedSuccessTime
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        X = []
        for i in range(machine_numbers):
            X.append([
                len(my_history_choice),
                my_push_distribute[i],
                agent1AdjustPush[i],
                my_reward_distribute[i],
                my_reward_distribute[i]/my_push_distribute[i] if my_push_distribute[i]!=0 else 0,
                agent1AdjustMachineReward[i]/agent1AdjustPush[i] if agent1AdjustPush[i]!=0 else 0,
                opp_push_distribute,
                agent2AdjustPush
            ])
        X = np.array(X)
        print(X.shape)
        y = self.model.predict(X)
        # print(y.shape)
        index = 0
        max = y[index]
        for i in range(len(y)):
            if y[i] > max:
                max = y[i]
                index = i
        return index

if __name__ == '__main__':
    bot = agent()