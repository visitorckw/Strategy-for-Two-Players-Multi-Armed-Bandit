import math

class agent():
    def __init__(self, c = math.sqrt(2)):
        self.c = c
        self.delta = 1e-9
    def likely(self, prob, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        p = prob
        ans = 1
        for i in range(max(len(my_history_choice), len(opp_history_choice))):
            if agent == 1:
                if i < len(my_history_choice):
                    if my_history_choice[i] == machine:
                        if my_history_reward[i] == 1:
                            ans *= p
                        else:
                            ans *= (1 - p)
                        p *= 0.97
                if i < len(opp_history_choice):
                    if opp_history_choice[i] == machine:
                        p *= 0.97
            else:
                if i < len(opp_history_choice):
                    if opp_history_choice[i] == machine:
                        p *= 0.97
                if i < len(my_history_choice):
                    if my_history_choice[i] == machine:
                        if my_history_reward[i] == 1:
                            ans *= p
                        else:
                            ans *= (1 - p)
                        p *= 0.97
        return ans
    def dfdx(self, x, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        y = self.likely(x, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
        y1 = self.likely(x+self.delta, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
        return (y1 - y) / self.delta
    def binary_search(self, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        delta = 1e-9
        L = 0
        R = 1 - delta
        while L < R:
            dL = self.dfdx(L, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            dR = self.dfdx(R, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if dL * dR >= 0:
                break
            M = (L + R) / 2
            dM = self.dfdx(M, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if abs(dM) <= self.delta:
                L = R = M
                break
            if dM > 0:
                L = M + self.delta
            else:
                R = M - self.delta
        return (L + R) / 2
    def linear_search(self, agent, machine, my_history_choice, opp_history_choice, my_history_reward):
        N = 100
        maxLikely = -1
        ans = -1
        for i in range(N+1):
            p = i / N
            like = self.likely(p, agent, machine, my_history_choice, opp_history_choice, my_history_reward)
            if like > maxLikely:
                maxLikely = like
                ans = p
        return ans
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
        self.prob = [ 0 for i in range(machine_numbers)]
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
            ev = self.linear_search(agent, i, my_history_choice, opp_history_choice, my_history_reward)
            self.prob[i] = ev
            ucb = ev + self.c * math.sqrt(math.log(N) / n[i])
            if ucb > maxUCB:
                maxUCB = ucb
                choice = i
        return choice