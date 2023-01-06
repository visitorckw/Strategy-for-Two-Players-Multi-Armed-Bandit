import statsmodels.api as sm
import numpy as np


class agent():
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

        if len(my_history_choice) < machine_numbers * 3:
            return len(my_history_choice) % machine_numbers
        
        predict = [0 for i in range(machine_numbers)]
        x = [[] for i in range(machine_numbers)]
        for i in range(len(my_history_choice)):
            choice = my_history_choice[i]
            x[choice].append(my_history_reward[i])

        choice = -1
        maxPredict = -1e9
        for i in range(machine_numbers):
            # print('fit data length:', len(x))
            # print('x = ', x)
            model = sm.tsa.arima.ARIMA(x[i], order=(1, 1, 1))
            fitted = model.fit()
            y = fitted.forecast(1)
            # print('y = ', y)
            predict[i] = y[0]
            if predict[i] > maxPredict:
                choice = i
                maxPredict = predict[i]
        return choice


if __name__ == '__main__':
    x = [[i+1] for i in range(2)]
    y = [2 * i + 5 for i in range(10)]
    y = np.array(y)
    # x = [1,1,0,1,1,0]
    x = [0,1,0]
    test = [1,2,3,2]

    model = sm.tsa.arima.ARIMA(x, order=(1, 1, 1))
    fitted = model.fit()
    print(fitted.summary())
    print(y)
    print(fitted.forecast(1))