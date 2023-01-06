from sklearn.linear_model import LinearRegression
import numpy as np

class agent():
    def __init__(self) -> None:
        self.predict = []
        pass
    def play(self, data):
        if len(self.predict) == 0:
            self.predict = [0.5 for i in range(data.machine_numbers)]
        if len(data.my_history_choice) > 0:
            last_choice = data.my_history_choice[-1]
            last_reward = data.my_history_reward[-1]
            model = LinearRegression()
            x = []
            y = []
            for i in range(len(data.my_history_choice)):
                if data.my_history_choice[i] != last_choice:
                    continue
                y.append([data.my_history_reward[i]])
            for i in range(len(y)):
                x.append([i+1])
            if len(x) and len(y):
                model.fit(x, y)
                ev = model.predict([[len(x)]])
                self.predict[last_choice] = ev[0][0]
        chose = -1
        maxi = -1
        for i in range(data.machine_numbers):
            if self.predict[i] >= maxi:
                maxi = self.predict[i]
                chose = i
        return chose