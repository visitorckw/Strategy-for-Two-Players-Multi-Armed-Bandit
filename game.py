import random

class game:
    def __init__(self, N = 100, gameRounds = 2000, random_seed = 10):
        random.seed(random_seed)
        self.N = N # 機器的數量
        self.machine = [] # 每一台機器當前得到糖果的機器
        self.gameRounds = gameRounds # 遊戲總共進行幾輪
        self.round = 0 # 當前進行到第幾輪
        self.agent1Reward = 0 # 當前agent1所獲得的糖果總數量
        self.agent2Reward = 0 # 當前agent2所獲得的糖果總數量
        self.historyAgent1Choice = [] # 過去每一輪Agent1所選擇的機器編號
        self.historyAgent1Reward = [] # 過去每一輪Agent1所獲得的糖果
        self.historyAgent2Choice = [] # 過去每一輪Agent1所選擇的機器編號
        self.historyAgent2Reward = [] # 過去每一輪Agent1所獲得的糖果
        for i in range(self.N):
            self.machine.append(random.random())
    def agent1Play(self, choice, log):
        if 0 > choice or choice >= self.N:
            print("INVALID CHOICE BY AGENT1 !!!")
            print('choice = ', choice)
            exit(1)
        reward = 1 if random.random() <= self.machine[choice] else 0
        self.agent1Reward += reward
        self.historyAgent1Choice.append(choice)
        self.historyAgent1Reward.append(reward)
        self.machine[choice] *= 0.97
        self.round += 1
        if log:
            print('agent1 choice', choice, 'get reward', reward, 'total score', str(self.agent1Reward) + ' vs ' + str(self.agent2Reward))
        return reward
    def agent2Play(self, choice, log):  
        if 0 > choice or choice >= self.N:
            print("INVALID CHOICE BY AGENT2 !!!")
            print('choice = ', choice)
            exit(1)
        reward = 1 if random.random() <= self.machine[choice] else 0
        self.agent2Reward += reward
        self.historyAgent2Choice.append(choice)
        self.historyAgent2Reward.append(reward)
        self.machine[choice] *= 0.97
        self.round += 1
        if log:
            print('agent2 choice', choice, 'get reward', reward, 'total score', str(self.agent1Reward) + ' vs ' + str(self.agent2Reward))
        return reward
    def undoAgent1(self):
        choice = self.historyAgent1Choice[-1]
        reward = self.historyAgent1Reward[-1]
        self.agent1Reward -= reward
        self.machine[choice] /= 0.97
        self.historyAgent1Choice.pop()
        self.historyAgent1Reward.pop()
        self.round -= 1
    def undoAgent2(self):
        choice = self.historyAgent2Choice[-1]
        reward = self.historyAgent2Reward[-1]
        self.agent2Reward -= reward
        self.machine[choice] /= 0.97
        self.historyAgent2Choice.pop()
        self.historyAgent2Reward.pop()
        self.round -= 1
    
    def run(self, agent1, agent2, log = False):
        while self.round < self.gameRounds:
            if self.round % 2 == 0: # 輪到 agent1行動
                # play(agent, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward)
                choice = agent1.play(1, self.N, self.gameRounds, self.round, self.agent1Reward, self.historyAgent1Choice, self.historyAgent2Choice, self.historyAgent1Reward) # agnet1 的play函數應該要return所選的機器編號
                self.agent1Play(choice, log)
            else:
                choice = agent2.play(2, self.N, self.gameRounds, self.round, self.agent2Reward, self.historyAgent2Choice, self.historyAgent1Choice, self.historyAgent2Reward) # agnet1 的play函數應該要return所選的機器編號
                self.agent2Play(choice, log)
        print(self.agent1Reward, ':', self.agent2Reward)
        if self.agent1Reward == self.agent2Reward:
            if log:
                print('GAME TIED')
            return 0
        if self.agent1Reward > self.agent2Reward:
            if log:
                print('AGENT1 WIN')
            return 1
        else:
            if log:
                print('AGENT2 WIN')
            return -1