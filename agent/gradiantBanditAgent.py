import random
import math
import numpy as np

class agent():
    def __init__(self) -> None:
        self.H = []
        self.pi = []
        self.alpha = 0.5

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

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
        
        if len(self.H) == 0:
            self.H = [0 for i in range(machine_numbers)]
        if len(my_history_choice) > 0:
            ## update H and pi
            last_choice = my_history_choice[-1]
            last_reward = my_history_reward[-1]
            for i in range(machine_numbers):
                avg_reward = 0.5
                if my_push_distribute[i]:
                    my_reward_distribute[i] / my_push_distribute[i]
                if i == last_choice:
                    self.H[i] = self.H[i] + self.alpha * (last_reward - avg_reward) * (1 - self.pi[i])
                else:
                    self.H[i] = self.H[i] - self.alpha * (last_reward - avg_reward) * self.pi[i]
        self.pi = self.softmax(self.H)
        return np.random.choice(machine_numbers, p=self.pi)
        

if __name__ == '__main__':
    arr = [1,1,1]
    a = agent()
    print(a.softmax(arr))
