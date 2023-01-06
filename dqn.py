import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import numpy as np
import random
from game import game as g
from game import Struct
from helper import plot
from agent import randomAgent


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear2 = nn.Linear(input_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name='dqn.pt'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * \
                    torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        # print(loss)
        self.optimizer.step()


class Agent:
    def __init__(self):
        self.epsilon = 0
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(450, 225, 50)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.curstate
        state = np.squeeze(state).reshape(-1)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):

        self.epsilon = 80 - self.n_games
        final_move = [0 for i in range(50)]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 49)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # print(prediction.shape)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = g(dataCollect=True, dataName=None)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        # print(final_move)
        # perform move and get new state
        reward, done, score = game.agent1Play(final_move)
        agent2 = randomAgent.agent()
        data = {
            "agent": 2,
            "machine_numbers": game.N,
            "total_round": game.gameRounds,
            "current_round": game.round,
            "my_total_rewards": game.agent2Reward,
            "my_history_choice": game.historyAgent2Choice,
            "opp_history_choice": game.historyAgent1Choice,
            "my_history_reward": game.historyAgent2Reward,
            "my_push_distribute": game.agent2Push,
            "opp_push_distribute": game.agent1Push,
            "my_reward_distribute": game.agent2MachineReward,
            "adjustedChoose": game.agent2AdjustPush,
            "opponentAdjustedChoose": game.agent1AdjustPush,
            "adjustedSuccessTime": game.agent2AdjustMachineReward,
        }
        choice = agent2.play(Struct(**data))
        game.agent2Play(choice)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game = g(dataCollect=True, dataName=None)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
