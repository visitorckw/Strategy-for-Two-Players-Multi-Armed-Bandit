import math

class agent():
    def __init__(self, c = math.sqrt(2)):
        self.c = c
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
        w = [0 for i in range(machine_numbers)]
        n = [0 for i in range(machine_numbers)]
        N = len(my_history_choice)
        for i in range(len(my_history_choice)):
            n[my_history_choice[i]] += 1
            w[my_history_choice[i]] += my_history_reward[i]
        maxUCB = -1
        choice = -1
        for i in range(machine_numbers):
            ucb = w[i] / n[i] + self.c * math.sqrt(math.log(N) / n[i])
            if ucb > maxUCB:
                maxUCB = ucb
                choice = i
        return choice