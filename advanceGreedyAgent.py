import time
class agent():
    def __init__(self):
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
                        prob *= 0.97
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

    def play(self, agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute):
        if len(my_history_choice) < machine_numbers: # 尚未嘗試過所有機器
            for i in range(machine_numbers):
                if i not in my_history_choice:
                    return i
        # push_times = [0 for i in range(machine_numbers)]
        # for i in range(len(my_history_choice)):
        #     push_times[my_history_choice[i]] += 1
        # for i in range(len(opp_history_choice)):
        #     push_times[opp_history_choice[i]] += 1
        expectProb = []
        for i in range(machine_numbers):
            # t = time.time()
            prob = self.linear_search(agent, i, my_history_choice, opp_history_choice, my_history_reward)
            # print(time.time() - t)
            expectProb.append(prob * (0.97 ** (my_push_distribute[i] + opp_push_distribute[i])))
        # print('advance', expectProb)
        choice = -1
        maxProfit = -1
        for i in range(machine_numbers):
            if expectProb[i] > maxProfit:
                maxProfit = expectProb[i]
                choice = i
        return choice