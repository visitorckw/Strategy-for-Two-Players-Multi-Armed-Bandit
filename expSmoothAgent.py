class agent():
    def __init__(self, alpha = 0.5) -> None:
        self.predict = []
        self.alpha = alpha
    def play(self, data):
        if len(self.predict) == 0:
            self.predict = [0.5 for i in range(data.machine_numbers)]
        if len(data.my_history_choice) > 0:
            last_choice = data.my_history_choice[-1]
            last_reward = data.my_history_reward[-1]
            self.predict[last_choice] = (1 - self.alpha) * self.predict[last_choice] + self.alpha * last_reward
        chose = -1
        maxi = -1
        for i in range(data.machine_numbers):
            if self.predict[i] >= maxi:
                maxi = self.predict[i]
                chose = i
        return chose