import random
import numpy as np
import pandas as pd
import os
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
def calLoss(act, pred):
    act = np.array(act)
    pred = np.array(pred)
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff

class game:
    def __init__(self, N = 100, gameRounds = 2000, random_seed = -1, dataCollect = False, dataName = "TrainData"):
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
        self.agent1AdjustMachineReward = [0 for i in range(N)] 
        self.agent2AdjustMachineReward = [0 for i in range(N)] 
        self.agent1AdjustPush = [0 for i in range(N)] 
        self.agent2AdjustPush = [0 for i in range(N)] 
        self.dataCollect = dataCollect
        self.dataName = dataName
        self.collectQuantile = [0.25,0.5,0.75,0.9]
        self.curstate = [[0 for i in range(9)] for j in range(50)]
        if(self.dataCollect):
            self.data = pd.DataFrame()
        for i in range(self.N):
            self.machine.append(random.random())
        self.initProb = self.machine
    def agent1Play(self, choice, log=False):
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
        self.agent1AdjustMachineReward[choice]=self.agent1AdjustMachineReward[choice]*0.97+reward
        self.agent2AdjustMachineReward[choice]=self.agent2AdjustMachineReward[choice]*0.97+reward
        self.agent1AdjustPush[choice] = self.agent1AdjustPush[choice]*0.97+1
        self.agent2AdjustPush[choice] = self.agent2AdjustPush[choice]*0.97+1
        self.round += 1
        if log:
            print('agent1 choice', choice, 'get reward', reward, 'total score', str(self.agent1Reward) + ' vs ' + str(self.agent2Reward))
        return reward, self.round==self.N-1, self.agent1Reward
    def agent2Play(self, choice, log=False):  
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
        self.agent1AdjustMachineReward[choice]=self.agent1AdjustMachineReward[choice]*0.97+reward
        self.agent2AdjustMachineReward[choice]=self.agent2AdjustMachineReward[choice]*0.97+reward
        self.agent1AdjustPush[choice] = self.agent1AdjustPush[choice]*0.97+1
        self.agent2AdjustPush[choice] = self.agent2AdjustPush[choice]*0.97+1
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
                data = {
                    "agent": 1,
                    "machine_numbers" : self.N,
                    "total_round" : self.gameRounds,
                    "current_round" : self.round,
                    "my_total_rewards" : self.agent1Reward,
                    "my_history_choice" : self.historyAgent1Choice,
                    "opp_history_choice" : self.historyAgent2Choice, 
                    "my_history_reward" : self.historyAgent1Reward,
                    "my_push_distribute" : self.agent1Push,
                    "opp_push_distribute" : self.agent2Push,
                    "my_reward_distribute" : self.agent1MachineReward,
                    "adjustedChoose": self.agent1AdjustPush,
                    "opponentAdjustedChoose": self.agent2AdjustPush,
                    "adjustedSuccessTime": self.agent1AdjustMachineReward,
                }
                # play(agent, machine_numbers, total_round, current_round, my_total_rewards, my_history_choice, opp_history_choice, my_history_reward, my_push_distribute, opp_push_distribute, my_reward_distribute)
                choice = agent1.play(Struct(**data)) # agnet1 的play函數應該要return所選的機器編號
                self.agent1Play(choice, log)
            else:
                data = {
                    "agent": 2,
                    "machine_numbers" : self.N,
                    "total_round" : self.gameRounds,
                    "current_round" : self.round,
                    "my_total_rewards" : self.agent2Reward,
                    "my_history_choice" : self.historyAgent2Choice,
                    "opp_history_choice" : self.historyAgent1Choice, 
                    "my_history_reward" : self.historyAgent2Reward,
                    "my_push_distribute" : self.agent2Push,
                    "opp_push_distribute" : self.agent1Push,
                    "my_reward_distribute" : self.agent2MachineReward,
                    "adjustedChoose": self.agent2AdjustPush,
                    "opponentAdjustedChoose": self.agent1AdjustPush,
                    "adjustedSuccessTime": self.agent2AdjustMachineReward,
                }
                choice = agent2.play(Struct(**data)) # agnet2 的play函數應該要return所選的機器編號
                self.agent2Play(choice, log)
            # if agent1.prob!=None:
            #     print("Agent1 loss: ",calLoss(self.initProb , agent1.prob))
            if self.dataCollect:
                for q in self.collectQuantile:
                    if self.round == self.gameRounds*q:
                        self.curstate = []
                        for i in range(self.N):
                            curstate = {
                                "originalProb":self.initProb[i],
                                "step":self.round,
                                "chooseTime": self.agent1Push[i],
                                "adjustedChoose": self.agent1AdjustPush[i],
                                "successTime": self.agent1MachineReward[i],
                                "adjustedSuccessTime": self.agent1AdjustMachineReward[i],
                                "hitRate": self.agent1MachineReward[i]/self.agent1Push[i] if self.agent1Push[i]!=0 else 0,
                                "adjustedHitRate": self.agent1AdjustMachineReward[i]/self.agent1AdjustPush[i] if self.agent1AdjustPush[i]!=0 else 0,
                                "opponentChooseTime": self.agent2Push[i],
                                "adjustedOpponentChooseTime": self.agent2AdjustPush[i],
                            }
                            self.curstate.append([
                                curstate["step"],
                                curstate["chooseTime"],
                                curstate["adjustedChoose"],
                                curstate["successTime"],
                                curstate["adjustedSuccessTime"],
                                curstate["hitRate"],
                                curstate["adjustedHitRate"],
                                curstate["adjustedOpponentChooseTime"],
                            ])
                            self.data = self.data.append(curstate,ignore_index = True)
        print(self.agent1Reward, ':', self.agent2Reward)
        if self.dataCollect and self.dataName!=None:
            self.data.to_csv(self.dataName,mode='a', header=not os.path.exists(self.dataName), index=False)
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