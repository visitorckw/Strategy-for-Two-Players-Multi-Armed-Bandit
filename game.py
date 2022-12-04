import random
import numpy as np
def calLoss(act, pred):
    act = np.array(act)
    pred = np.array(pred)
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff

class game:
    def __init__(self, N = 100, gameRounds = 2000, random_seed = -1):
        if random_seed != -1:
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
        self.agent1MachineReward = [0 for i in range(N)] # Agnet1從每一台機器上的總獲利
        self.agent2MachineReward = [0 for i in range(N)] # Agnet2從每一台機器上的總獲利
        self.agent1Push = [0 for i in range(N)] # Agnet1每一台機器玩過的次數
        self.agent2Push = [0 for i in range(N)] # Agnet1每一台機器玩過的次數
        for i in range(self.N):
            self.machine.append(random.random())
        self.initProb = self.machine
    def agent1Play(self, choice, log):
        if 0 > choice or choice >= self.N:
            print("INVALID CHOICE BY AGENT1 !!!")
            print('choice = ', choice)
            exit(1)
        reward = 1 if random.random() <= self.machine[choice] else 0
        self.agent1Reward += reward
        self.historyAgent1Choice.append(choice)
        self.historyAgent1Reward.append(reward)
        self.agent1MachineReward[choice] += reward
        self.agent1Push[choice] += 1
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
        self.agent2MachineReward[choice] += reward
        self.agent2Push[choice] += 1
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
        self.agent1MachineReward[choice] -= reward
        self.agent1Push[choice] -= 1
        self.round -= 1
    def undoAgent2(self):
        choice = self.historyAgent2Choice[-1]
        reward = self.historyAgent2Reward[-1]
        self.agent2Reward -= reward
        self.machine[choice] /= 0.97
        self.historyAgent2Choice.pop()
        self.historyAgent2Reward.pop()
        self.agent2MachineReward[choice] -= reward
        self.agent2Push[choice] -= 1
        self.round -= 1
    
    def run(self, agent1, agent2, log = False):
        while self.round < self.gameRounds:
            if self.round % 2 == 0: # 輪到 agent1行動
                # play(agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute)
                choice = agent1.play(1, self.N, self.gameRounds, self.round, self.agent1Reward, self.historyAgent1Choice, self.historyAgent2Choice, self.historyAgent1Reward, self.agent1Push, self.agent2Push, self.agent1MachineReward) # agnet1 的play函數應該要return所選的機器編號
                self.agent1Play(choice, log)
            else:
                choice = agent2.play(2, self.N, self.gameRounds, self.round, self.agent2Reward, self.historyAgent2Choice, self.historyAgent1Choice, self.historyAgent2Reward, self.agent2Push, self.agent1Push, self.agent2MachineReward) # agnet1 的play函數應該要return所選的機器編號
                self.agent2Play(choice, log)
            if agent1.prob!=None:
                print("Agent1 loss: ",calLoss(self.initProb , agent1.prob))
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