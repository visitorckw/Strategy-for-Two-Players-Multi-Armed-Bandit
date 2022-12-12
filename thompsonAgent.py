from typing import Tuple
import copy
import numpy as np
class agent():
    def __init__(self,prior: Tuple[int, int] = (10, 10)) -> None:
        self.prob = None
        self.distr = {"prior": prior, "data": []}
        self.arms_beliefs = None

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
        if self.arms_beliefs is None:
            # Init priors on the first step
            self.arms_beliefs = {k: copy.deepcopy(self.distr) 
                                 for k in range(machine_numbers)}
        else:
            if len(my_history_reward) > 0:
                self.arms_beliefs[my_history_choice[-1]]['data'].append(my_history_reward[-1])
        pvals = []
        for key, distr in self.arms_beliefs.items():
            # Observed counts
            observed_a = sum(distr["data"])
            observed_b = len(distr["data"]) - observed_a
            # Add prior counts
            beta_a = observed_a + distr["prior"][0]
            beta_b = observed_b + distr["prior"][1]

            # Draw p for Bernoulli distribution from Beta distribution
            pvals.append((key, np.random.beta(beta_a, beta_b)))
        # print(max(pvals, key=lambda d: d[1])[0])
        return max(pvals, key=lambda d: d[1])[0]