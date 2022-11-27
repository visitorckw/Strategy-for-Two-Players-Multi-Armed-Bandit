class agent():
    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute):
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        avg_rewards = [0 for i in range(machine_numbers)]
        push_times = [0 for i in range(machine_numbers)]
        for i in range(len(my_history_choice)):
            push_times[my_history_choice[i]] += 1
            avg_rewards[my_history_choice[i]] += my_history_reward[i]
        for i in range(machine_numbers):
            avg_rewards[i] /= push_times[i]
        choice = -1
        maxProfit = -1
        for i in range(machine_numbers):
            if avg_rewards[i] > maxProfit:
                maxProfit = avg_rewards[i]
                choice = i
        return choice